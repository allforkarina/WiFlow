# Merge Real GT Files and Separate Reference Keypoints

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `build_memmap.py` read real MM-Fi ground truth files and merge them in Axx→Syy order, while preserving rgb-derived keypoints as `reference_keypoints.npy`. The model architecture stays OpenPose18 for now (will be updated in a later step).

**Architecture:** Fix multiprocessing to produce deterministic trial order, add `--gt-dir` argument, read GT files in the same trial order as CSI/RGB data, extract x,y channels (drop depth), save as `ground_truth.npy`. Keep existing rgb-derived keypoints as `reference_keypoints.npy`.

**Tech Stack:** Python, NumPy, scipy (existing), multiprocessing

---

## Context

The current `scripts/build_memmap.py` treats `rgb/frame*.npy` keypoints as ground truth and saves them as `ground_truth.npy`. In reality, these are COCO17 reference coordinates — the real MM-Fi ground truth is in `ground_truth_npy/`.

### GT Data Format

- **1080 files**: `E{env}_S{subj}_A{action}.npy` — each is one trial's GT
- **Shape**: `(N_frames, 17, 3)` — (x, y, depth) per keypoint, H36M-17 topology
- **Coordinate range**: x,y in [-0.8, 0.8]; depth values up to ~3.3
- **Environments**: E01→S01-S10, E02→S11-S20, E03→S21-S30, E04→S31-S40
- **Actions**: A01-A27 (27 actions)
- **All files**: 297 frames per trial (consistent)
- **Paths**: Windows `C:\Users\LuvRene\Downloads\ground_truth_npy`, Linux `/data/WiFiPose/dataset/ground_truth_npy`

### Current build_memmap.py Issues

1. **Non-deterministic trial ordering**: `ProcessPoolExecutor.as_completed()` collects results in completion order, not submission order. This means CSI frames and keypoints are in a non-deterministic sequence across runs.
2. **Wrong GT source**: rgb-derived keypoints are saved as `ground_truth.npy` but they are reference only.
3. **GT file naming**: GT files are named `E{env}_{subject}_{action}.npy`. The env is derivable from subject via `derive_env()`: `int(subj[1:]) → (n-1)//10+1`.

---

## Changes Required

### 1. `scripts/build_memmap.py` — modifications

**1a. Fix deterministic trial ordering**

Replace `as_completed` collection with ordered collection. After all futures complete, collect results in the original `trials` order using a dict keyed by trial path. This ensures CSI, reference keypoints, and GT frames are all aligned.

**1b. Add `--gt-dir` argument**

Path to the `ground_truth_npy/` directory. Required parameter.

**1c. Read and merge real GT files in trial order**

After trial processing and before/after concatenating CSI and reference keypoints, iterate trials in order:

```python
for trial in trials:  # ordered iteration
    action = trial.parent.name   # e.g., "A01"
    subject = trial.name          # e.g., "S01"
    env = derive_env(subject)     # e.g., "E01"
    gt_file = gt_dir / f"{env}_{subject}_{action}.npy"
    gt_data = np.load(str(gt_file))           # (N, 17, 3)
    gt_kpts = gt_data[..., :2].copy()          # (N, 17, 2) — drop depth
    all_gt.append(gt_kpts)
```

Then concatenate: `np.concatenate(all_gt, axis=0)` → `ground_truth.npy` with shape `(N_total, 17, 2)`.

**1d. Rename rgb keypoints output**

Save rgb-derived keypoints as `reference_keypoints.npy` (shape `(N_total, 18, 2)`, OpenPose18 format) instead of `ground_truth.npy`.

**1e. Update stats.json**

Add entries documenting the new file layout.

---

## Files NOT Modified

- `data/memmap_dataset.py` — loads `ground_truth.npy`, will need update in later step when model switches to H36M-17
- `dataloader.py` — `kpts18` key stays for now, will change in later step
- `train.py`, `eval.py`, model files — unchanged

---

## Verification

1. **Run build_memmap.py** on a small subset:
   ```bash
   python scripts/build_memmap.py --src /data/WiFiPose/dataset/dataset \
       --dst /tmp/test_memmap --gt-dir /data/WiFiPose/dataset/ground_truth_npy \
       --train-subjects S01 --workers 1
   ```

2. **Verify output shapes and alignment:**
   ```python
   import numpy as np
   gt = np.load("/tmp/test_memmap/ground_truth.npy")        # (N, 17, 2)
   ref = np.load("/tmp/test_memmap/reference_keypoints.npy") # (N, 18, 2)
   csi = np.load("/tmp/test_memmap/csi_gminmax.npy")         # (N, 64, 3, 114)
   assert gt.shape[0] == ref.shape[0] == csi.shape[0]
   assert gt.shape[1:] == (17, 2)  # H36M-17, x,y only
   ```

3. **Verify GT coordinate range** is within [-0.8, 0.8]:
   ```python
   assert gt.min() >= -0.85 and gt.max() <= 0.85
   ```

4. **Verify deterministic**: Run twice with same `--workers 4`, diff the output files — they should be identical.
