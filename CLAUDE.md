# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

WiFlow: WiFi CSI-based human pose estimation using H36M-17 keypoints.

**Skeleton topology**: H36M-17 (17 joints, 16 bone edges). Defined in `models/skeleton.py` as `H36M_BONE_EDGES` and `H36M17_NAMES`. Pelvis is index 0, thorax is 8, shoulders are 11 (left) and 14 (right), hips are 1 (right) and 4 (left).

**Coordinate system**: pose_range `[-0.8, 0.8]` for both x and y. GT files in `ground_truth_npy/` are `(N, 17, 3)` — (x, y, confidence). The third channel (confidence/depth) is dropped during memmap build.

**Normalization**: CSI is stored in 3 pre-computed variants — `csi_gminmax.npy`, `csi_gzscore.npy`, `csi_zscore.npy`. Training uses `global_minmax` by default.

## Commands

```bash
# Build memmap from raw MM-Fi dataset
python scripts/build_memmap.py \
    --src /data/WiFiPose/dataset/dataset \
    --dst /data/WiFiPose/dataset/mmfi_pose_v3 \
    --gt-dir /data/WiFiPose/dataset/ground_truth_npy \
    --train-subjects S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 \
    --workers 8

# Train
python train.py --dataset-root data/mmfi_pose --decoder-type joint --epochs 50

# Evaluate
python eval.py --dataset-root data/mmfi_pose --checkpoint outputs/train/best_val_mpjpe.pth

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_skeleton.py -v

# Diagnose GT file (print coords + visualize a frame)
python scripts/diagnose_gt.py --gt-file /path/to/E01_S01_A01.npy --frame 0

# Diagnose data pipeline and initial loss
python scripts/diagnose_loss.py --dataset-dir data/mmfi_pose
```

## Architecture

**Data flow**: Raw MM-Fi (`wifi-csi/frame*.mat` + `rgb/frame*.npy`) → `scripts/build_memmap.py` → memmap `.npy` files → `data/memmap_dataset.py` (mmap read) → `dataloader.py` (collate) → `train.py` / `eval.py`

**Output files from build_memmap**:
| File | Format | Purpose |
|------|--------|---------|
| `ground_truth.npy` | `(N, 17, 2)` H36M-17 | Training labels |
| `reference_keypoints.npy` | `(N, 17, 2)` raw COCO17 | Reference only, not for training |
| `csi_gminmax.npy` etc. | `(N, 64, 3, 114)` | Pre-normalized CSI |
| `meta.npz` | environment, sample, action, frame_idx | Split building |

**Model**: `WiFlowModel` = `WiFlowSpatialEncoder` (CNN) → `WiFlowAxialEncoder` (attention) → decoder. Three decoder types:

- `joint` — 17 learnable query vectors + cross-attention to spatial tokens + GNN refinement. Output `(B, 17, 2)`.
- `hierarchical` — 2-stage coarse-to-fine joint retrieval. Stage 0: torso core (9 joints), Stage 1: limb ends (8 joints). Output `(B, 17, 2)`.
- `heatmap_msfn` — Multi-stage PCM/PAF decoder. 3 stages with PAPM modulation. Output dict with `keypoints` and `stages`. PAF channels = 32 (2 × 16 edges). PCM channels = 17.

Axial encoder modes: `spatial_then_temporal`, `temporal_then_spatial`, `parallel_sum`, `parallel_concat`.

**Loss**: `joint`/`hierarchical` → `L1(coord) + 0.5 × L1(bone_length)`. `heatmap_msfn` → `MSE(pcm) + MSE(paf)` over all stages.

**Training**: AdamW(`lr=2e-5`, `wd=5e-4`) + OneCycleLR(`max_lr=5e-4`, 30% warmup, cosine anneal). Gradient clipping at 1.0. PCK normalized by torso scale (right_shoulder to left_hip).

**Deprecated files**: `data/heatmap_gt.py` — pre-H36M17 offline PCM/PAF generation. Use `pose_targets.py` instead.

## Key conventions

- All keypoints in pose_range `[-0.8, 0.8]`, NOT `[0, 1]`
- Heatmap coordinate mapping: `(x + 0.8) / 1.6 × (H-1)` and reverse
- `data_dir` parameter for memmap dataset paths (NOT `dataset_root`)
- File-by-file H36M-17 migration is in progress — some downstream files may still reference old OpenPose18 names
