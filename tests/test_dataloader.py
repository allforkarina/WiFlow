from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from dataloader import (
    CSI_SHAPE,
    DEFAULT_SPLIT_SCHEME,
    KEYPOINT_SHAPE,
    MMFiPoseDataset,
    RAW_STORAGE_ATTR,
    SPLIT_NAMES,
    SampleSequence,
    FrameRecord,
    build_frame_splits,
    build_h5_dataset,
    build_sample_splits,
    summarize_splits,
)


def _make_sample_sequences() -> list[SampleSequence]:
    return [
        SampleSequence(
            action="A01",
            sample=f"S{index:02d}",
            environment="env1",
            rgb_dir=Path(f"rgb_{index:02d}"),
            csi_dir=Path(f"csi_{index:02d}"),
        )
        for index in range(1, 11)
    ]


def _make_frame_record(sample: str, frame_stem: str = "frame001") -> FrameRecord:
    return FrameRecord(
        action="A01",
        sample=sample,
        environment="env1",
        frame_stem=frame_stem,
        keypoint_path=Path(f"{sample}_{frame_stem}.npy"),
        csi_path=Path(f"{sample}_{frame_stem}.mat"),
    )


def _write_dual_scheme_h5(path: Path) -> None:
    keypoints = np.zeros((3, *KEYPOINT_SHAPE), dtype=np.float32)
    keypoints[0, :, 0] = 10.0
    keypoints[0, :, 1] = 20.0
    keypoints[1, :, 0] = 5.0
    keypoints[1, :, 1] = 10.0
    keypoints[2, :, 0] = 2.0
    keypoints[2, :, 1] = 4.0

    amplitude = np.zeros((3, *CSI_SHAPE), dtype=np.float32)
    amplitude[0] = 10.0
    amplitude[1] = 5.0
    amplitude[2] = 2.0
    phase = np.zeros((3, *CSI_SHAPE), dtype=np.float32)
    phase_cos = np.ones((3, *CSI_SHAPE), dtype=np.float32)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("keypoints", data=keypoints)
        h5_file.create_dataset("csi_amplitude", data=amplitude)
        h5_file.create_dataset("csi_phase", data=phase)
        h5_file.create_dataset("csi_phase_cos", data=phase_cos)
        h5_file.create_dataset("action", data=np.array(["A01", "A01", "A01"], dtype=string_dtype))
        h5_file.create_dataset("sample", data=np.array(["S01", "S02", "S03"], dtype=string_dtype))
        h5_file.create_dataset("environment", data=np.array(["env1", "env1", "env1"], dtype=string_dtype))
        h5_file.create_dataset("frame_id", data=np.array(["frame001", "frame002", "frame003"], dtype=string_dtype))
        h5_file.create_dataset("train_indices", data=np.array([0], dtype=np.int64))
        h5_file.create_dataset("val_indices", data=np.array([1], dtype=np.int64))
        h5_file.create_dataset("test_indices", data=np.array([2], dtype=np.int64))
        h5_file.create_dataset("action_env_train_indices", data=np.array([0], dtype=np.int64))
        h5_file.create_dataset("action_env_val_indices", data=np.array([1], dtype=np.int64))
        h5_file.create_dataset("action_env_test_indices", data=np.array([2], dtype=np.int64))
        h5_file.create_dataset("frame_random_train_indices", data=np.array([0], dtype=np.int64))
        h5_file.create_dataset("frame_random_val_indices", data=np.array([1], dtype=np.int64))
        h5_file.create_dataset("frame_random_test_indices", data=np.array([2], dtype=np.int64))
        h5_file.attrs["storage_format"] = RAW_STORAGE_ATTR
        h5_file.attrs["keypoint_normalization"] = "train_axis_max"
        h5_file.attrs["amplitude_normalization"] = "train_global_minmax"
        h5_file.attrs["keypoint_x_scale"] = 10.0
        h5_file.attrs["keypoint_y_scale"] = 20.0
        h5_file.attrs["amplitude_train_min"] = 0.0
        h5_file.attrs["amplitude_train_max"] = 10.0
        h5_file.attrs["action_env_keypoint_x_scale"] = 10.0
        h5_file.attrs["action_env_keypoint_y_scale"] = 20.0
        h5_file.attrs["action_env_amplitude_train_min"] = 0.0
        h5_file.attrs["action_env_amplitude_train_max"] = 10.0
        h5_file.attrs["frame_random_keypoint_x_scale"] = 20.0
        h5_file.attrs["frame_random_keypoint_y_scale"] = 40.0
        h5_file.attrs["frame_random_amplitude_train_min"] = 0.0
        h5_file.attrs["frame_random_amplitude_train_max"] = 20.0


def _write_legacy_h5(path: Path) -> None:
    keypoints = np.ones((1, *KEYPOINT_SHAPE), dtype=np.float32)
    amplitude = np.full((1, *CSI_SHAPE), 0.5, dtype=np.float32)
    phase = np.zeros((1, *CSI_SHAPE), dtype=np.float32)
    phase_cos = np.ones((1, *CSI_SHAPE), dtype=np.float32)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("keypoints", data=keypoints)
        h5_file.create_dataset("csi_amplitude", data=amplitude)
        h5_file.create_dataset("csi_phase", data=phase)
        h5_file.create_dataset("csi_phase_cos", data=phase_cos)
        h5_file.create_dataset("action", data=np.array(["A01"], dtype=string_dtype))
        h5_file.create_dataset("sample", data=np.array(["S01"], dtype=string_dtype))
        h5_file.create_dataset("environment", data=np.array(["env1"], dtype=string_dtype))
        h5_file.create_dataset("frame_id", data=np.array(["frame001"], dtype=string_dtype))
        h5_file.create_dataset("train_indices", data=np.array([0], dtype=np.int64))
        h5_file.create_dataset("val_indices", data=np.array([], dtype=np.int64))
        h5_file.create_dataset("test_indices", data=np.array([], dtype=np.int64))
        h5_file.attrs["keypoint_normalization"] = "train_axis_max"
        h5_file.attrs["keypoint_x_scale"] = 10.0
        h5_file.attrs["keypoint_y_scale"] = 20.0
        h5_file.attrs["amplitude_normalization"] = "train_global_minmax"
        h5_file.attrs["amplitude_train_min"] = 0.0
        h5_file.attrs["amplitude_train_max"] = 1.0


def test_build_sample_splits_keeps_each_sample_in_one_split(monkeypatch) -> None:
    sequences = _make_sample_sequences()
    monkeypatch.setattr("dataloader.discover_sample_sequences", lambda dataset_root: sequences)

    splits = build_sample_splits("unused", seed=42)

    assert len(splits["train"]) == 6
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 2
    assigned_samples = {
        split_name: {sequence.sample for sequence in split_sequences}
        for split_name, split_sequences in splits.items()
    }
    assert assigned_samples["train"].isdisjoint(assigned_samples["val"])
    assert assigned_samples["train"].isdisjoint(assigned_samples["test"])
    assert assigned_samples["val"].isdisjoint(assigned_samples["test"])


def test_build_frame_splits_is_deterministic_and_disjoint(monkeypatch) -> None:
    sequences = _make_sample_sequences()
    records = [_make_frame_record(sequence.sample) for sequence in sequences]
    monkeypatch.setattr("dataloader.discover_sample_sequences", lambda dataset_root: sequences)
    monkeypatch.setattr("dataloader.expand_frame_records", lambda split_sequences: records)

    first = build_frame_splits("unused", seed=42)
    second = build_frame_splits("unused", seed=42)

    assert len(first["train"]) == 6
    assert len(first["val"]) == 2
    assert len(first["test"]) == 2
    assert [record.sample for record in first["train"]] == [record.sample for record in second["train"]]
    assert {record.sample for record in first["train"]}.isdisjoint({record.sample for record in first["val"]})
    assert {record.sample for record in first["train"]}.isdisjoint({record.sample for record in first["test"]})


def test_mmfi_pose_dataset_normalizes_raw_data_per_split_scheme(tmp_path) -> None:
    dataset_path = tmp_path / "dual_scheme.h5"
    _write_dual_scheme_h5(dataset_path)

    action_env_dataset = MMFiPoseDataset(dataset_root=dataset_path, split="train", split_scheme="action_env")
    frame_random_dataset = MMFiPoseDataset(dataset_root=dataset_path, split="train", split_scheme="frame_random")

    action_env_sample = action_env_dataset[0]
    frame_random_sample = frame_random_dataset[0]

    assert np.allclose(action_env_sample["keypoints"], 1.0)
    assert np.allclose(action_env_sample["csi_amplitude"], 1.0)
    assert np.allclose(frame_random_sample["keypoints"], 0.5)
    assert np.allclose(frame_random_sample["csi_amplitude"], 0.5)


def test_mmfi_pose_dataset_supports_legacy_action_env_only(tmp_path) -> None:
    dataset_path = tmp_path / "legacy.h5"
    _write_legacy_h5(dataset_path)

    dataset = MMFiPoseDataset(dataset_root=dataset_path, split="train")
    sample = dataset[0]

    assert dataset.split_scheme == DEFAULT_SPLIT_SCHEME
    assert np.allclose(sample["keypoints"], 1.0)
    assert np.allclose(sample["csi_amplitude"], 0.5)

    with pytest.raises(KeyError):
        MMFiPoseDataset(dataset_root=dataset_path, split="train", split_scheme="frame_random")


def test_summarize_splits_uses_requested_scheme(tmp_path) -> None:
    dataset_path = tmp_path / "dual_scheme.h5"
    _write_dual_scheme_h5(dataset_path)

    summary = summarize_splits(dataset_path, split_scheme="frame_random")

    assert tuple(summary.keys()) == SPLIT_NAMES
    assert summary["train"]["num_frames"] == 1
    assert summary["val"]["num_frames"] == 1
    assert summary["test"]["num_frames"] == 1


def test_build_h5_dataset_writes_dual_split_indices(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "packed.h5"
    sequences = _make_sample_sequences()
    records_by_sample = {
        sequence.sample: FrameRecord(
            action=sequence.action,
            sample=sequence.sample,
            environment=sequence.environment,
            frame_stem="frame001",
            keypoint_path=Path(f"{sequence.sample}.npy"),
            csi_path=Path(f"{sequence.sample}.mat"),
        )
        for sequence in sequences
    }

    monkeypatch.setattr("dataloader.resolve_dataset_root", lambda dataset_root: Path("unused_root"))
    monkeypatch.setattr("dataloader.discover_sample_sequences", lambda dataset_root: sequences)
    monkeypatch.setattr(
        "dataloader.build_sample_splits",
        lambda dataset_root, seed=42, split_ratios=None: {
            "train": sequences[:6],
            "val": sequences[6:8],
            "test": sequences[8:],
        },
    )
    monkeypatch.setattr(
        "dataloader.build_frame_splits",
        lambda dataset_root, seed=42, split_ratios=None: {
            "train": [records_by_sample[sequence.sample] for sequence in sequences[:6]],
            "val": [records_by_sample[sequence.sample] for sequence in sequences[6:8]],
            "test": [records_by_sample[sequence.sample] for sequence in sequences[8:]],
        },
    )
    monkeypatch.setattr(
        "dataloader.expand_frame_records",
        lambda split_sequences: [records_by_sample[sequence.sample] for sequence in split_sequences],
    )

    def fake_prepare_keypoints_and_amplitude(record: FrameRecord) -> tuple[np.ndarray, np.ndarray]:
        keypoints = np.ones(KEYPOINT_SHAPE, dtype=np.float32) * float(int(record.sample[1:]))
        amplitude = np.ones(CSI_SHAPE, dtype=np.float32) * float(int(record.sample[1:]))
        return keypoints, amplitude

    def fake_prepare_raw_frame(record: FrameRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        keypoints, amplitude = fake_prepare_keypoints_and_amplitude(record)
        phase = np.zeros(CSI_SHAPE, dtype=np.float32)
        phase_cos = np.ones(CSI_SHAPE, dtype=np.float32)
        return keypoints, amplitude, phase, phase_cos

    monkeypatch.setattr("dataloader._prepare_keypoints_and_amplitude", fake_prepare_keypoints_and_amplitude)
    monkeypatch.setattr("dataloader._prepare_raw_frame", fake_prepare_raw_frame)

    summary = build_h5_dataset("unused", output_path, seed=42)

    assert summary["num_records"] == 10
    with h5py.File(output_path, "r") as h5_file:
        assert "train_indices" in h5_file
        assert "action_env_train_indices" in h5_file
        assert "frame_random_train_indices" in h5_file
        assert h5_file.attrs["storage_format"] == RAW_STORAGE_ATTR
        assert h5_file.attrs["default_split_scheme"] == DEFAULT_SPLIT_SCHEME
        assert "frame_random_keypoint_x_scale" in h5_file.attrs
