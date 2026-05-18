"""
Build memory-mapped .npy files from MM-Fi dataset for fast training I/O.

Pre-computes 3 normalization variants and stores each as a single .npy file.
Training loader uses np.load(path, mmap_mode='r') for zero-copy OS-cached access.

Input:
    /data/WiFiPose/dataset/dataset/{ACTION}/{SUBJECT}/
        wifi-csi/frame*.mat   ← CSIamp (3, 114, 10)
        rgb/frame*.npy        ← COCO17 keypoints (17, 2) — reference only

Output:
    csi_gminmax.npy          ← (N, 64, 3, 114)
    csi_gzscore.npy          ← (N, 64, 3, 114)
    csi_zscore.npy           ← (N, 64, 3, 114)
    ground_truth.npy         ← H36M-17 GT (N, 17, 2)
    reference_keypoints.npy  ← raw COCO17 (N, 17, 2) — not for training
    meta.npz
    stats.json

Usage:
    python scripts/build_memmap.py \
        --src /data/WiFiPose/dataset/dataset \
        --dst /data/WiFiPose/dataset/mmfi_pose_v3 \
        --gt-dir /data/WiFiPose/dataset/ground_truth_npy \
        --train-subjects S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 \
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import resample


TIME_PACKETS = 64
RX_ANTENNAS = 3
SUBCARRIERS = 114


def derive_env(subject: str) -> str:
    num = int(subject.lstrip("S"))
    return f"env{(num - 1) // 10 + 1}"


def sanitize_csi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if finite.all():
        return x
    fill = float(np.median(x[finite])) if finite.any() else 0.0
    return np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill).astype(np.float32)


def preprocess_csi_one_frame(csi_amp: np.ndarray) -> np.ndarray:
    csi_amp = sanitize_csi(np.asarray(csi_amp, dtype=np.float32))
    csi_amp = sanitize_csi(resample(csi_amp, TIME_PACKETS, axis=-1))
    return np.transpose(csi_amp, (2, 0, 1)).astype(np.float32, copy=False)


def iter_trials(src_root: Path) -> list[Path]:
    trials: list[Path] = []
    for action_dir in sorted(p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("A")):
        for subj_dir in sorted(p for p in action_dir.iterdir() if p.is_dir() and p.name.startswith("S")):
            if (subj_dir / "wifi-csi").is_dir() and (subj_dir / "rgb").is_dir():
                trials.append(subj_dir)
    return trials


def _parse_frame_number(stem: str) -> int:
    if not stem.startswith("frame") or not stem[5:].isdigit():
        raise ValueError(f"Expected frame filename pattern 'frame<num>', got: {stem}")
    return int(stem[5:])


def process_trial(trial_dir: Path) -> dict | None:
    action = trial_dir.parent.name
    subject = trial_dir.name
    wifi_dir = trial_dir / "wifi-csi"
    rgb_dir = trial_dir / "rgb"

    mat_paths = sorted(wifi_dir.glob("frame*.mat"))
    npy_paths = sorted(rgb_dir.glob("frame*.npy"))
    if not mat_paths:
        return None

    mat_stems = {p.stem: p for p in mat_paths}
    npy_stems = {p.stem: p for p in npy_paths}
    common = sorted(set(mat_stems) & set(npy_stems), key=_parse_frame_number)
    if not common:
        return None

    n_frames = len(common)
    csi_frames = np.empty((n_frames, TIME_PACKETS, RX_ANTENNAS, SUBCARRIERS), dtype=np.float32)
    reference_kpts = np.zeros((n_frames, 17, 2), dtype=np.float32)
    frame_idx = np.zeros(n_frames, dtype=np.int64)

    for i, stem in enumerate(common):
        mat = sio.loadmat(str(mat_stems[stem]))
        csi_frames[i] = preprocess_csi_one_frame(np.asarray(mat["CSIamp"], dtype=np.float32))
        reference_kpts[i] = np.load(str(npy_stems[stem]))
        frame_idx[i] = _parse_frame_number(stem)

    return {
        "csi": csi_frames,
        "reference_keypoints": reference_kpts,
        "environment": derive_env(subject),
        "sample": subject,
        "action": action,
        "frame_idx": frame_idx,
    }


def _worker(args):
    trial_dir = args
    try:
        result = process_trial(Path(trial_dir))
        label = f"{Path(trial_dir).parent.name}/{Path(trial_dir).name}"
        return label, result, None
    except Exception:
        label = f"{Path(trial_dir).parent.name}/{Path(trial_dir).name}"
        return label, None, traceback.format_exc()


def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def main():
    parser = argparse.ArgumentParser(description="Build memmap .npy files from MM-Fi dataset")
    parser.add_argument("--src", default="/data/WiFiPose/dataset/dataset")
    parser.add_argument("--dst", default="/data/WiFiPose/dataset/mmfi_pose_v3")
    parser.add_argument("--train-subjects", nargs="+",
                        default=["S01","S02","S03","S04","S05","S06","S07","S08","S09","S10",
                                 "S11","S12","S13","S14","S15","S16","S17","S18","S19","S20"])
    parser.add_argument("--pose-min", type=float, default=-0.8)
    parser.add_argument("--pose-max", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gt-dir", default="", help="Path to ground_truth_npy/ directory")
    args = parser.parse_args()
    if not args.gt_dir:
        parser.error("--gt-dir is required (path to ground_truth_npy/)")

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)
    train_set = set(args.train_subjects)

    trials = iter_trials(src_root)
    print(f"Found {len(trials)} trials")

    all_data: list[dict] = []
    trial_results: dict[str, dict] = {}
    failures: list[tuple[str, str]] = []
    t0 = time.time()

    if args.workers <= 1:
        for i, trial in enumerate(trials):
            label = f"{trial.parent.name}/{trial.name}"
            result = process_trial(trial)
            if result:
                trial_results[str(trial)] = result
                print(f"  [{i+1}/{len(trials)}] {label} ok N={result['csi'].shape[0]}")
            else:
                print(f"  [{i+1}/{len(trials)}] {label} SKIP")
        for trial in trials:
            key = str(trial)
            if key in trial_results:
                all_data.append(trial_results[key])
    else:
        tasks = [str(t) for t in trials]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_worker, t): t for t in tasks}
            done = 0
            for fut in as_completed(futs):
                done += 1
                label, result, err = fut.result()
                if err:
                    failures.append((label, err))
                    print(f"  [{done}/{len(trials)}] {label} FAIL")
                elif result:
                    trial_path = futs[fut]
                    trial_results[trial_path] = result
                    print(f"  [{done}/{len(trials)}] {label} ok N={result['csi'].shape[0]}")
                else:
                    print(f"  [{done}/{len(trials)}] {label} SKIP")
        for trial in trials:
            key = str(trial)
            if key in trial_results:
                all_data.append(trial_results[key])

    dt = time.time() - t0
    print(f"\nPhase 1: {len(all_data)} ok, {len(failures)} fail ({dt:.1f}s)")
    if not all_data:
        print("ERROR: No trials processed")
        sys.exit(1)

    print("Concatenating...")
    all_csi_raw  = np.concatenate([d["csi"] for d in all_data], axis=0).astype(np.float32)
    all_reference_keypoints = np.concatenate([d["reference_keypoints"] for d in all_data], axis=0).astype(np.float32)
    all_envs     = np.array([e for d in all_data for e in [d["environment"]] * d["csi"].shape[0]])
    all_subjects = np.array([s for d in all_data for s in [d["sample"]]      * d["csi"].shape[0]])
    all_actions  = np.array([a for d in all_data for a in [d["action"]]      * d["csi"].shape[0]])
    all_fidx     = np.concatenate([d["frame_idx"] for d in all_data])

    n_total = all_csi_raw.shape[0]
    n_train = int(np.isin(all_subjects.astype(str), list(train_set)).sum())
    print(f"Total: {n_total}, train: {n_train} ({n_train/n_total*100:.1f}%)")

    print("Loading ground truth...")
    gt_dir = Path(args.gt_dir)
    all_gt_parts = []
    for trial in trials:
        if str(trial) not in trial_results:
            continue
        action = trial.parent.name
        subject = trial.name
        env_num = (int(subject.lstrip("S")) - 1) // 10 + 1
        gt_file = gt_dir / f"E{env_num:02d}_{subject}_{action}.npy"
        gt_data = np.load(str(gt_file))
        gt_kpts = gt_data[..., :2].copy()
        expected_frames = trial_results[str(trial)]["csi"].shape[0]
        assert gt_kpts.shape[0] == expected_frames, \
            f"Frame mismatch: {gt_file.name} has {gt_kpts.shape[0]}, expected {expected_frames}"
        all_gt_parts.append(gt_kpts.astype(np.float32))
    all_gt = np.concatenate(all_gt_parts, axis=0)
    print(f"  ground truth: {all_gt.shape}")

    print("Computing normalization statistics (train set only)...")
    train_mask = np.isin(all_subjects.astype(str), list(train_set))
    train_csi = all_csi_raw[train_mask]
    amp_min  = float(train_csi.min())
    amp_max  = float(train_csi.max())
    amp_mean = float(train_csi.mean())
    amp_std  = float(train_csi.std())
    print(f"  min={amp_min:.4f}  max={amp_max:.4f}  mean={amp_mean:.4f}  std={amp_std:.4f}")

    print("  global_minmax...")
    csi_gminmax = safe_div(all_csi_raw - amp_min, amp_max - amp_min).astype(np.float32)

    print("  global_zscore...")
    csi_gzscore = safe_div(all_csi_raw - amp_mean, amp_std).astype(np.float32)

    print("  zscore...")
    csi_zscore = np.empty_like(all_csi_raw, dtype=np.float32)
    for i in range(n_total):
        x = all_csi_raw[i]
        m, s = float(x.mean()), float(x.std())
        csi_zscore[i] = safe_div(x - m, s).astype(np.float32)

    del all_csi_raw

    print("Saving .npy files...")
    t_save = time.time()

    np.save(str(dst_root / "csi_gminmax.npy"), csi_gminmax)
    np.save(str(dst_root / "csi_gzscore.npy"), csi_gzscore)
    np.save(str(dst_root / "csi_zscore.npy"),  csi_zscore)
    np.save(str(dst_root / "reference_keypoints.npy"), all_reference_keypoints)
    np.save(str(dst_root / "ground_truth.npy"), all_gt)
    np.savez(str(dst_root / "meta.npz"),
             environment=all_envs, sample=all_subjects,
             action=all_actions, frame_idx=all_fidx)

    stats = {
        "amplitude_train_min": amp_min,
        "amplitude_train_max": amp_max,
        "amplitude_train_mean": amp_mean,
        "amplitude_train_std": amp_std,
        "time_packets": TIME_PACKETS,
        "rx_antennas": RX_ANTENNAS,
        "subcarriers": SUBCARRIERS,
        "total_frames": n_total,
        "train_frames": n_train,
        "normalization_subjects": args.train_subjects,
        "pose_format": "H36M17",
        "reference_format": "raw_coco17_no_mapping",
        "gt_source": str(args.gt_dir),
        "ground_truth_shape": list(all_gt.shape),
        "reference_keypoints_shape": list(all_reference_keypoints.shape),
    }
    with open(dst_root / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    total_mb = sum(
        f.stat().st_size for f in dst_root.glob("*.npy") if f.is_file()
    ) / (1024 * 1024)
    print(f"Done in {time.time()-t_save:.0f}s — {total_mb:.0f} MB total .npy files")


if __name__ == "__main__":
    main()