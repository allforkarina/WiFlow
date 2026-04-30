# Repository Guidelines

## Project Structure & Module Organization
- `dataloader.py`: Core module for discovering samples, packing HDF5 files, loading splits, creating PyTorch `DataLoader` instances, and previewing split contents.
- One packed HDF5 can now hold both `action_env` and `frame_random` split schemes; training and evaluation default to `action_env` and can switch with `--split-scheme`.
- `models/`: PyTorch model code, including the full WiFlow model, CSI spatial encoder, axial attention encoder, attention pooler, skeleton-aware decoder, and shared COCO17 skeleton topology. The active model path is CSI feature concat -> spatial encoder -> axial encoder -> attention pooler -> skeleton-aware decoder.
- `train.py`: Root-level training entrypoint for WiFlow pose regression, including losses, metrics, optimizer, scheduler, checkpointing, and CSV logging.
- `eval.py`: Root-level evaluation entrypoint for loading checkpoints, computing test metrics, and saving CSI/skeleton visualizations.
- `scripts/build_h5_dataset.py`: Command-line wrapper that builds a single `.h5`/`.hdf5` dataset from the raw MM-Fi directory structure.
- `tests/`: `pytest` unit tests. Mirror module names such as `tests/test_dataloader.py`, `tests/test_wiflow_model.py`, or `tests/test_wiflow_decoder.py`.
- `.gitignore`: Excludes Python caches, local environments, generated datasets, checkpoints, and editor files from Git.

Generated datasets can be large and should not be committed. Keep raw dataset roots outside the repository.

## Project Domain Knowledge
- One CSI sample is a physical signal tensor shaped `3 antennas x 114 subcarriers x 10 frames`. The subcarrier axis carries spatial-frequency response, the antenna axis carries spatial phase-difference and direction information, and the temporal axis carries motion cues such as Doppler effects.
- CSI amplitude and phase are complementary physical quantities with different noise patterns. When both are used, do not assume they should be processed identically or fused by blind concatenation.
- The target pose is the structured COCO17 keypoint set. The 17 joints are not independent coordinates; they are constrained by the human skeleton topology.
- The central modeling gap is that CSI is a low-resolution, high-noise, implicit sensing signal, while pose regression needs precise coordinates. Strong skeleton priors are important for bridging that gap.
- Preserve CSI physical dimension semantics where practical. Avoid arbitrary flattening or pooling that mixes antenna, subcarrier, and temporal meanings before the model has selected useful information.
- Prefer attention-based information selection over destructive pooling for low-SNR CSI features, and use structured supervision such as bone or topology-aware losses in addition to coordinate losses.

## Build, Test, and Development Commands
Use the existing Conda environment for development commands:

```powershell
conda activate WiFiPose
pip install numpy scipy h5py tqdm torch pytest
```

Build an HDF5 dataset:

```powershell
python scripts\build_h5_dataset.py --dataset-root D:\path\to\raw\dataset --output-path data\mmfi_pose.h5 --seed 42
```

Inspect an HDF5 dataset:

```powershell
python dataloader.py --dataset-root data\mmfi_pose.h5 --preview
```

Run tests:

```powershell
pytest
```

Run a quick training sanity check:

```powershell
python train.py --dataset-root data\mmfi_pose.h5 --epochs 5 --subset-size 32 --output-dir outputs\sanity
```

Run the default training configuration:

```powershell
python train.py --dataset-root data\mmfi_pose.h5 --epochs 50 --batch-size 64 --output-dir outputs\train
```

The default training configuration uses `csi_amplitude,csi_phase_cos` input features, `OneCycleLR`, gradient clipping, `coord_l1 + 0.5 * bone_l1`, the baseline axial mode `spatial_then_temporal`, and stronger AdamW weight decay than the original baseline. Use `--csi-features csi_amplitude` for an amplitude-only run.

Run an axial-attention encoder ablation:

```powershell
python train.py --dataset-root data\mmfi_pose.h5 --axial-mode temporal_then_spatial --epochs 50 --batch-size 64 --output-dir outputs\train_temporal_then_spatial
```

Supported `--axial-mode` values are `spatial_then_temporal`, `temporal_then_spatial`, `parallel_sum`, and `parallel_concat`. Checkpoints store the selected mode in `train_config`, and evaluation rebuilds the model from that saved configuration.

Run the frame-random split configuration:

```powershell
python train.py --dataset-root data\mmfi_pose.h5 --split-scheme frame_random --epochs 50 --batch-size 64 --output-dir outputs\train_frame_random
```

Evaluate one checkpoint:

```powershell
python eval.py --dataset-root data\mmfi_pose.h5 --checkpoint outputs\train\best_val_mpjpe.pth --output-dir outputs\eval
```

Evaluate against the frame-random split:

```powershell
python eval.py --dataset-root data\mmfi_pose.h5 --checkpoint outputs\train_frame_random\best_val_mpjpe.pth --split-scheme frame_random --output-dir outputs\eval_frame_random
```

## Coding Style & Naming Conventions
Use Python 3.10+ syntax, type hints, and `pathlib.Path` for paths. Group imports as standard library, third-party, then local. Follow existing naming: `snake_case` functions/variables, `PascalCase` classes, and uppercase constants such as `SPLIT_NAMES`. Use 4-space indentation. Keep comments focused on dataset assumptions, shapes, and normalization.

## Testing Guidelines
Automated tests use `pytest`. Add tests for split generation, path validation, shape validation, normalization edge cases, model shape contracts, and HDF5 round-tripping. Name files `test_*.py` and tests `test_<behavior>()`. Use temporary directories and tiny synthetic fixtures.

Training and evaluation outputs are written under `outputs/` by default. Checkpoints include `best_val_mpjpe.pth`, `best_val_pck_0_2.pth`, and `last.pth`; epoch metrics are appended to `train_log.csv`. Evaluation visualizations are saved as `.png` files grouped by action/environment samples.

## Commit & Pull Request Guidelines
This checkout has no `.git` history, so no convention can be inferred. Use concise imperative commits, for example `Add HDF5 split preview`. Pull requests should include a summary, commands run, dataset assumptions, and relevant shape or frame-count output. Do not commit generated datasets, virtual environments, or machine-specific paths.

## Security & Configuration Tips
Do not hard-code private dataset locations beyond documented defaults. Pass dataset paths with `--dataset-root` and keep large or sensitive data outside version control.

## Agent-Specific Instructions
Write repository-facing agent notes, documentation, and code comments in English. Keep comments neatly aligned with surrounding style. Use Chinese for conversational replies unless the user requests another language.

Whenever project code changes, update this `AGENTS.md` file in the same turn if the change affects commands, structure, conventions, testing, configuration, or agent workflow.

After each project modification, commit the change and push it to the configured GitHub remote in the same turn unless the user explicitly asks not to push.

Before changing code, apply the `karpathy-guidelines` skill: state assumptions when needed, prefer the smallest working change, avoid unrelated refactors, and verify the result with a concrete check.

Before running project code or tests, activate the existing Conda environment with `conda activate WiFiPose` to ensure commands run in the established project environment.
