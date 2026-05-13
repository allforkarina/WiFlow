from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dataloader import (
    SPLIT_NAMES,
    create_memmap_data_loader,
    create_memmap_data_loaders,
    memmap_collate_fn,
)


def _make_temp_memmap_dataset(tmp_path: Path, num_samples: int = 30) -> Path:
    data_dir = tmp_path / "memmap_dataset"
    data_dir.mkdir()

    csi = np.random.randn(num_samples, 64, 3, 114).astype(np.float32)
    np.save(str(data_dir / "csi_gminmax.npy"), csi)

    kpts18 = np.random.randn(num_samples, 18, 2).astype(np.float32)
    np.save(str(data_dir / "ground_truth.npy"), kpts18)

    subjects = []
    for i in range(num_samples):
        subjects.append(f"S{i // 3:02d}")
    meta = {
        "environment": np.array([f"env{i % 3}" for i in range(num_samples)]),
        "sample": np.array(subjects),
        "action": np.array([f"A{i % 5:02d}" for i in range(num_samples)]),
    }
    np.savez(str(data_dir / "meta.npz"), **meta)

    return data_dir


def test_memmap_collate_fn_produces_expected_shapes() -> None:
    batch = [
        {
            "csi": torch.randn(64, 3, 114),
            "kpts18": torch.randn(18, 2),
            "meta": {"action": "A01", "subject": "S01", "env": "env1", "frame_idx": 0},
        }
        for _ in range(4)
    ]

    result = memmap_collate_fn(batch)

    assert result["csi_amplitude"].shape == (4, 3, 114, 64)
    assert result["keypoints"].shape == (4, 18, 2)
    assert len(result["action"]) == 4
    assert len(result["sample"]) == 4
    assert len(result["environment"]) == 4
    assert len(result["frame_idx"]) == 4


def test_create_memmap_data_loader_returns_expected_batch(tmp_path) -> None:
    data_dir = _make_temp_memmap_dataset(tmp_path)

    loader = create_memmap_data_loader(
        data_dir=data_dir,
        split="train",
        batch_size=4,
        num_workers=0,
        seed=42,
    )

    batch = next(iter(loader))

    assert batch["csi_amplitude"].shape == (4, 3, 114, 64)
    assert batch["keypoints"].shape == (4, 18, 2)
    assert len(batch["action"]) == 4
    assert len(batch["sample"]) == 4
    assert len(batch["environment"]) == 4
    assert len(batch["frame_idx"]) == 4


def test_create_memmap_data_loaders_returns_all_splits(tmp_path) -> None:
    data_dir = _make_temp_memmap_dataset(tmp_path)

    loaders = create_memmap_data_loaders(
        data_dir=data_dir,
        batch_size=4,
        num_workers=0,
        seed=42,
    )

    assert set(loaders.keys()) == set(SPLIT_NAMES)
    for split in SPLIT_NAMES:
        batch = next(iter(loaders[split]))
        assert batch["csi_amplitude"].ndim == 4
        assert batch["keypoints"].ndim == 3


def test_create_memmap_data_loader_rejects_invalid_split(tmp_path) -> None:
    data_dir = _make_temp_memmap_dataset(tmp_path)

    try:
        create_memmap_data_loader(
            data_dir=data_dir,
            split="invalid",
            batch_size=4,
        )
    except ValueError as exc:
        assert "split must be one of" in str(exc)
    else:
        raise AssertionError("Expected create_memmap_data_loader to reject invalid split")


def test_create_memmap_data_loader_train_shuffles_by_default(tmp_path) -> None:
    data_dir = _make_temp_memmap_dataset(tmp_path)

    train_loader = create_memmap_data_loader(
        data_dir=data_dir,
        split="train",
        batch_size=4,
        num_workers=0,
        seed=42,
    )
    val_loader = create_memmap_data_loader(
        data_dir=data_dir,
        split="val",
        batch_size=4,
        num_workers=0,
        seed=42,
    )

    assert train_loader.sampler is not None
    assert val_loader.sampler is not None


def test_memmap_collate_fn_preserves_metadata() -> None:
    batch = [
        {
            "csi": torch.randn(64, 3, 114),
            "kpts18": torch.randn(18, 2),
            "meta": {
                "action": f"A{i:02d}",
                "subject": f"S{i:02d}",
                "env": f"env{i % 3}",
                "frame_idx": i,
            },
        }
        for i in range(3)
    ]

    result = memmap_collate_fn(batch)

    assert result["action"] == ["A00", "A01", "A02"]
    assert result["sample"] == ["S00", "S01", "S02"]
    assert result["environment"] == ["env0", "env1", "env2"]
    assert result["frame_idx"] == [0, 1, 2]