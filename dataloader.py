from __future__ import annotations

"""NPY memmap-backed dataloader for MM-Fi pose data.

Collates H36M-17 keypoints from ground_truth.npy.
Optionally collates raw COCO17 reference_keypoints.
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from data.memmap_dataset import MemmapDataset

SPLIT_NAMES = ("train", "val", "test")

DEFAULT_TRAIN_SUBJECTS: tuple[str, ...] = (
    "S01", "S02", "S03", "S04", "S05", "S06", "S07",
    "S11", "S12", "S13", "S14", "S15", "S16", "S17",
)
DEFAULT_VAL_SUBJECTS: tuple[str, ...] = (
    "S08", "S09", "S18", "S19",
)
DEFAULT_TEST_SUBJECTS: tuple[str, ...] = (
    "S10", "S20",
)
CROSS_DOMAIN_SUBJECTS: tuple[str, ...] = (
    "S21", "S22", "S23", "S24", "S25", "S26", "S27", "S28", "S29", "S30",
    "S31", "S32", "S33", "S34", "S35", "S36", "S37", "S38", "S39", "S40",
)
KNOWN_SUBJECTS: tuple[str, ...] = (
    "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10",
    "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20",
)
DEFAULT_SPLIT_MODE = "subject_env_7_2_1"


def memmap_collate_fn(batch: list[dict]) -> dict:
    csi = torch.stack([item["csi"] for item in batch])
    csi = csi.permute(0, 2, 3, 1).contiguous()
    keypoints = torch.stack([item["keypoints"] for item in batch])
    result: dict = {
        "csi_amplitude": csi,
        "keypoints": keypoints,
        "action": [item["meta"]["action"] for item in batch],
        "sample": [item["meta"]["subject"] for item in batch],
        "environment": [item["meta"]["env"] for item in batch],
        "frame_idx": [item["meta"]["frame_idx"] for item in batch],
    }
    if "reference_keypoints" in batch[0]:
        result["reference_keypoints"] = torch.stack([item["reference_keypoints"] for item in batch])
    return result


def create_memmap_data_loader(
    data_dir: str | Path,
    split: str,
    batch_size: int,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    seed: int = 42,
    load_reference: bool = False,
    split_mode: str = DEFAULT_SPLIT_MODE,
    known_subjects: tuple[str, ...] = KNOWN_SUBJECTS,
    train_subjects: tuple[str, ...] = DEFAULT_TRAIN_SUBJECTS,
    val_subjects: tuple[str, ...] = DEFAULT_VAL_SUBJECTS,
    test_subjects: tuple[str, ...] = DEFAULT_TEST_SUBJECTS,
    cross_domain_subjects: tuple[str, ...] = CROSS_DOMAIN_SUBJECTS,
) -> DataLoader:
    if split not in SPLIT_NAMES + ("cross_domain",):
        raise ValueError(f"split must be one of {SPLIT_NAMES + ('cross_domain',)}, got {split}")

    dataset = MemmapDataset(
        data_dir=data_dir,
        split=split,
        split_mode=split_mode,
        known_subjects=known_subjects,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        cross_domain_subjects=cross_domain_subjects,
        seed=seed,
        load_reference=load_reference,
    )

    should_shuffle = shuffle if shuffle is not None else split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        collate_fn=memmap_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def create_memmap_data_loaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    load_reference: bool = False,
    split_mode: str = DEFAULT_SPLIT_MODE,
    known_subjects: tuple[str, ...] = KNOWN_SUBJECTS,
    train_subjects: tuple[str, ...] = DEFAULT_TRAIN_SUBJECTS,
    val_subjects: tuple[str, ...] = DEFAULT_VAL_SUBJECTS,
    test_subjects: tuple[str, ...] = DEFAULT_TEST_SUBJECTS,
    cross_domain_subjects: tuple[str, ...] = CROSS_DOMAIN_SUBJECTS,
) -> dict[str, DataLoader]:
    return {
        split: create_memmap_data_loader(
            data_dir=data_dir,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            load_reference=load_reference,
            split_mode=split_mode,
            known_subjects=known_subjects,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
            cross_domain_subjects=cross_domain_subjects,
        )
        for split in SPLIT_NAMES
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NPY memmap dataloader preview")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to the NPY memmap dataset directory")
    parser.add_argument("--preview", action="store_true", help="Load one sample from each split and print its shapes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.dataset_root)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    for split in SPLIT_NAMES + ("cross_domain",):
        loader = create_memmap_data_loader(
            data_dir=data_dir, split=split, batch_size=1, num_workers=0, load_reference=bool(args.preview),
        )
        dataset = loader.dataset
        subjects = sorted(set(dataset._samples[i] for i in dataset.indices))
        print(f"{split}: {len(dataset)} samples, subjects={subjects}")

        if args.preview:
            sample = dataset[0]
            ref_info = ""
            if "reference_keypoints" in sample:
                ref_info = f", reference_keypoints={tuple(sample['reference_keypoints'].shape)}"
            print(f"  {split}_preview: csi={tuple(sample['csi'].shape)}, keypoints={tuple(sample['keypoints'].shape)}{ref_info}, meta={sample['meta']}")


if __name__ == "__main__":
    main()