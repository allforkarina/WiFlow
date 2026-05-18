from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


CSI_FILES = {
    "global_minmax": "csi_gminmax.npy",
    "global_zscore": "csi_gzscore.npy",
    "zscore": "csi_zscore.npy",
}


class MemmapDataset(Dataset):
    """Memory-mapped .npy dataset for fast training I/O.

    CSI is stored as 3 pre-normalized .npy files, read via np.load(mmap_mode='r').
    Keypoints and meta are small enough to load entirely into RAM at init.

    ground_truth.npy — H36M-17 GT (N, 17, 2), used as training labels.
    reference_keypoints.npy — raw COCO17 (N, 17, 2), reference only (optional).

    Two split modes are supported:

    - ``subject_env_7_2_1``: Selects frames by fixed subject lists for train/val/test.
      Requires ``train_subjects``, ``val_subjects``, ``test_subjects`` parameters.
      ``cross_domain`` split returns all frames (no subject filtering).

    - ``frame_random``: Randomly splits frames within ``known_subjects`` using
      ``random_val_ratio`` (default 0.2). Ignores train/val/test subject lists.

    No HDF5 overhead, no compression — OS page cache handles I/O.
    Multiple DataLoader workers share the same OS buffer cache (mmap MAP_SHARED).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        split_mode: str = "subject_env_7_2_1",
        known_subjects: Iterable[str] | None = None,
        train_subjects: Iterable[str] | None = None,
        val_subjects: Iterable[str] | None = None,
        test_subjects: Iterable[str] | None = None,
        cross_domain_subjects: Iterable[str] | None = None,
        random_val_ratio: float = 0.2,
        seed: int = 42,
        time_packets: int = 64,
        subcarrier_mode: str = "keep",
        normalize: str = "global_minmax",
        load_reference: bool = False,
    ) -> None:
        if split not in {"train", "val", "test", "all", "cross_domain"}:
            raise ValueError(f"split must be train/val/test/all/cross_domain, got {split}")
        self.split = split
        self.normalize = normalize
        self.load_reference = load_reference
        self.split_mode = split_mode
        self.known_subjects = list(known_subjects) if known_subjects else None
        self.train_subjects = list(train_subjects) if train_subjects else None
        self.val_subjects = list(val_subjects) if val_subjects else None
        self.test_subjects = list(test_subjects) if test_subjects else None
        self.cross_domain_subjects = list(cross_domain_subjects) if cross_domain_subjects else None
        self.random_val_ratio = random_val_ratio

        data_dir = Path(data_dir)

        if normalize not in CSI_FILES:
            raise ValueError(f"Unknown normalize mode: {normalize}, expected one of {list(CSI_FILES)}")

        self._csi = np.load(str(data_dir / CSI_FILES[normalize]), mmap_mode="r")

        self._keypoints = np.load(str(data_dir / "ground_truth.npy"))

        self._reference_keypoints: np.ndarray | None = None
        if load_reference:
            ref_path = data_dir / "reference_keypoints.npy"
            if ref_path.exists():
                self._reference_keypoints = np.load(str(ref_path))

        meta = np.load(str(data_dir / "meta.npz"), allow_pickle=True)
        self._envs = meta["environment"]
        self._samples = meta["sample"]
        self._actions = meta["action"]
        self._frame_idx = meta["frame_idx"]

        self.indices = self._build_split(
            split, self.split_mode,
            self.known_subjects, self.train_subjects, self.val_subjects, self.test_subjects,
            self.cross_domain_subjects,
            self.random_val_ratio, seed,
        )

    def _build_split(
        self,
        split: str,
        split_mode: str,
        known_subjects: list[str] | None,
        train_subjects: list[str] | None,
        val_subjects: list[str] | None,
        test_subjects: list[str] | None,
        cross_domain_subjects: list[str] | None,
        random_val_ratio: float,
        seed: int,
    ) -> np.ndarray:
        sample_list = [str(s) for s in self._samples]

        if split_mode == "subject_env_7_2_1":
            if split == "all":
                subjects = known_subjects or []
            elif split == "cross_domain":
                subjects = cross_domain_subjects or []
            elif split == "train":
                subjects = train_subjects or []
            elif split == "val":
                subjects = val_subjects or []
            elif split == "test":
                subjects = test_subjects or []
            else:
                raise ValueError(f"Unknown split: {split}")

            indices = []
            for i in range(len(self._samples)):
                if sample_list[i] in subjects:
                    indices.append(i)
            return np.asarray(sorted(indices), dtype=np.int64)

        # split_mode == "frame_random"
        subject_filter = set(known_subjects) if known_subjects else None
        candidate_indices = []
        for i in range(len(self._samples)):
            if subject_filter is not None and sample_list[i] not in subject_filter:
                continue
            candidate_indices.append(i)

        if split == "all":
            return np.asarray(sorted(candidate_indices), dtype=np.int64)

        rng = random.Random(seed)
        grouped: dict[str, list[int]] = {}
        for idx in candidate_indices:
            grouped.setdefault(sample_list[idx], []).append(idx)

        train_indices: list[int] = []
        val_indices: list[int] = []
        for subject, indices in sorted(grouped.items()):
            shuffled = indices[:]
            rng.shuffle(shuffled)
            pivot = int(round(len(shuffled) * (1.0 - random_val_ratio)))
            train_indices.extend(shuffled[:pivot])
            val_indices.extend(shuffled[pivot:])

        if split == "train":
            return np.asarray(sorted(train_indices), dtype=np.int64)
        else:
            return np.asarray(sorted(val_indices), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        frame_idx = int(self.indices[index])

        csi = np.array(self._csi[frame_idx])
        keypoints = self._keypoints[frame_idx].copy()

        item: dict = {
            "csi": torch.from_numpy(csi),
            "keypoints": torch.from_numpy(np.ascontiguousarray(keypoints)),
            "meta": {
                "env": str(self._envs[frame_idx]),
                "subject": str(self._samples[frame_idx]),
                "action": str(self._actions[frame_idx]),
                "frame_idx": int(self._frame_idx[frame_idx]),
            },
        }
        if self._reference_keypoints is not None:
            item["reference_keypoints"] = torch.from_numpy(
                np.ascontiguousarray(self._reference_keypoints[frame_idx].copy())
            )
        return item