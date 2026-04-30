from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from torch.utils.data import DataLoader

from dataloader import DEFAULT_SPLIT_SCHEME, SPLIT_SCHEMES, MMFiPoseDataset, denormalize_keypoints
from models import COCO_BONE_EDGES, WiFlowModel
from train import compute_metrics, compute_torso_scale, prepare_model_input, select_device


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device) -> tuple[WiFlowModel, tuple[str, ...]]:
    """Load a trained WiFlow model and its CSI feature configuration from a checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device)   # load the checkpoint
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing model_state_dict: {checkpoint_path}")
    train_config = checkpoint.get("train_config")
    if not isinstance(train_config, Mapping) or "csi_features" not in train_config:
        raise KeyError(f"Checkpoint is missing train_config.csi_features: {checkpoint_path}")

    csi_features = tuple(train_config["csi_features"])
    axial_mode = str(train_config.get("axial_mode", "spatial_then_temporal"))
    model = WiFlowModel(input_channels=len(csi_features) * 3, axial_mode=axial_mode).to(device)   # load the model
    model.load_state_dict(checkpoint["model_state_dict"])           # load the model weights
    model.eval()                                                    # eval mode
    return model, csi_features


def plot_skeleton(
    ax: Axes,
    keypoints: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    title: str,
    color: str,
) -> None:
    """Plot one 2D skeleton on a Matplotlib axis."""

    ax.scatter(keypoints[:, 0], keypoints[:, 1], c=color, s=10)     # plot keypoints
    for start, end in edges:                                        # plot bones (connection)
        ax.plot(
            [keypoints[start, 0], keypoints[end, 0]],
            [keypoints[start, 1], keypoints[end, 1]],
            color=color,
            linewidth=1.5,
        )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")


def safe_stem(*parts: object) -> str:
    """Build a filesystem-safe output stem from sample metadata."""

    raw = "_".join(str(part) for part in parts)
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in raw)


def update_metric_totals(
    totals: Dict[str, float],
    metrics: Mapping[str, torch.Tensor],
    batch_size: int,
) -> None:
    """Accumulate batch metrics weighted by sample count."""

    for name, value in metrics.items():
        totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size


def average_metrics(totals: Mapping[str, float], sample_count: int) -> Dict[str, float]:
    return {name: value / max(sample_count, 1) for name, value in totals.items()}


def compute_joint_errors(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return per-sample per-joint Euclidean errors."""

    return torch.linalg.vector_norm(prediction - target, dim=-1)


def compute_joint_pck(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return per-sample per-joint correctness at one PCK threshold."""

    errors = compute_joint_errors(prediction, target)
    scale = compute_torso_scale(target, eps=eps).unsqueeze(-1)
    return (errors < (scale * threshold)).float()


def update_group_metric_totals(
    totals: Dict[str, Dict[str, float]],
    group_keys: Sequence[str],
    joint_errors: torch.Tensor,
    joint_pck: torch.Tensor,
) -> None:
    """Accumulate mean MPJPE and PCK@0.2 for each sample group."""

    for index, group_key in enumerate(group_keys):
        group_total = totals.setdefault(group_key, {"count": 0.0, "mpjpe": 0.0, "pck_0_2": 0.0})
        group_total["count"] += 1.0
        group_total["mpjpe"] += float(joint_errors[index].mean().item())
        group_total["pck_0_2"] += float(joint_pck[index].mean().item())


def build_group_metric_rows(
    totals: Mapping[str, Mapping[str, float]],
    group_label: str,
) -> list[dict[str, float | int | str]]:
    """Convert grouped metric totals to CSV-friendly rows."""

    rows: list[dict[str, float | int | str]] = []
    for group_name in sorted(totals):
        group_total = totals[group_name]
        count = int(group_total["count"])
        rows.append(
            {
                group_label: group_name,
                "sample_count": count,
                "mpjpe": group_total["mpjpe"] / max(count, 1),
                "pck_0_2": group_total["pck_0_2"] / max(count, 1),
            }
        )
    return rows


def build_joint_metric_rows(
    joint_errors: Sequence[torch.Tensor],
    joint_pck: Sequence[torch.Tensor],
) -> list[dict[str, float | int]]:
    """Aggregate per-joint MPJPE and PCK@0.2 across the evaluated split."""

    all_joint_errors = torch.cat(list(joint_errors), dim=0)
    all_joint_pck = torch.cat(list(joint_pck), dim=0)
    rows: list[dict[str, float | int]] = []
    for joint_index in range(all_joint_errors.shape[1]):
        rows.append(
            {
                "joint_index": joint_index,
                "sample_count": int(all_joint_errors.shape[0]),
                "mpjpe": float(all_joint_errors[:, joint_index].mean().item()),
                "pck_0_2": float(all_joint_pck[:, joint_index].mean().item()),
            }
        )
    return rows


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write CSV rows with a header inferred from the first row."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_metric_breakdowns(
    model: WiFlowModel,
    loader: DataLoader,
    device: torch.device,
    csi_features: tuple[str, ...],
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    """Collect per-joint, per-action, and per-environment metrics for one split."""

    action_totals: Dict[str, Dict[str, float]] = {}
    environment_totals: Dict[str, Dict[str, float]] = {}
    joint_error_batches: list[torch.Tensor] = []
    joint_pck_batches: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            model_input, target = prepare_model_input(batch, device, csi_features)
            prediction = model(model_input)
            joint_errors = compute_joint_errors(prediction, target).detach().cpu()
            joint_pck = compute_joint_pck(prediction, target).detach().cpu()
            joint_error_batches.append(joint_errors)
            joint_pck_batches.append(joint_pck)
            update_group_metric_totals(action_totals, batch["action"], joint_errors, joint_pck)
            update_group_metric_totals(environment_totals, batch["environment"], joint_errors, joint_pck)

    return (
        build_joint_metric_rows(joint_error_batches, joint_pck_batches),
        build_group_metric_rows(action_totals, "action"),
        build_group_metric_rows(environment_totals, "environment"),
    )


def evaluate_model(
    model: WiFlowModel,
    loader: DataLoader,
    device: torch.device,
    csi_features: tuple[str, ...],
) -> Dict[str, float]:
    """Evaluate the model on one dataloader and return averaged metrics."""

    totals: Dict[str, float] = {}
    sample_count = 0

    with torch.no_grad():
        for batch in loader:
            model_input, target = prepare_model_input(batch, device, csi_features)    # get test data
            prediction = model(model_input)                             # model prediction
            metrics = compute_metrics(prediction, target)               # compute metrics for the batch
            batch_size = target.shape[0]
            sample_count += batch_size
            update_metric_totals(totals, metrics, batch_size)           # accumulate metrics weighted by batch size

    return average_metrics(totals, sample_count)                        # average the metrics


def save_visualizations(
    model: WiFlowModel,
    dataset: MMFiPoseDataset,
    output_dir: Path,
    device: torch.device,
    csi_features: tuple[str, ...],
    max_visualizations: int | None = None,
) -> int:
    """Save one CSI and skeleton visualization for each action/environment pair."""

    output_dir.mkdir(parents=True, exist_ok=True)
    visualized_pairs: set[tuple[str, str]] = set()

    for index in range(len(dataset)):
        sample = dataset[index] 
        action = str(sample["action"])                  # get action from the sample
        environment = str(sample["environment"])        # get environment from the sample  
        pair = (action, environment)                    # each pair sample one visualization

        if pair in visualized_pairs:
            continue
        if max_visualizations is not None and len(visualized_pairs) >= max_visualizations:
            break

        # sample and predict
        batch = {
            feature_name: torch.as_tensor(sample[feature_name], dtype=torch.float32).unsqueeze(0)
            for feature_name in csi_features
        }
        batch["keypoints"] = torch.as_tensor(sample["keypoints"], dtype=torch.float32).unsqueeze(0)
        model_input, _ = prepare_model_input(batch, device, csi_features)
        with torch.no_grad():
            prediction = model(model_input).detach().cpu().numpy()[0]

        # visualize the CSI heatmap and the predicted vs. ground truth skeletons
        target = np.asarray(sample["keypoints"], dtype=np.float32)
        target_denorm = denormalize_keypoints(
            target,
            dataset.keypoint_x_scale,
            dataset.keypoint_y_scale,
        )
        prediction_denorm = denormalize_keypoints(
            prediction,
            dataset.keypoint_x_scale,
            dataset.keypoint_y_scale,
        )

        fig, axes = plt.subplots(3, 1, figsize=(6, 12))
        csi_heatmap = np.asarray(sample["csi_amplitude"], dtype=np.float32).reshape(342, 10)
        axes[0].imshow(csi_heatmap, aspect="auto", cmap="jet")
        axes[0].set_title(
            f"CSI Amplitude Heatmap ({action} in {environment}; features={','.join(csi_features)})"
        )
        axes[0].set_ylabel("Sub-channels (Flattened)")
        axes[0].set_xlabel("Packet Window (T=10)")

        plot_skeleton(axes[1], target_denorm, COCO_BONE_EDGES, "Ground Truth", color="green")
        plot_skeleton(axes[2], prediction_denorm, COCO_BONE_EDGES, "WiFlow Prediction", color="red")

        fig.tight_layout()
        filename = safe_stem(action, environment, f"frame{sample['frame_id']}") + ".png"
        fig.savefig(output_dir / filename)
        plt.close(fig)

        visualized_pairs.add(pair)

    return len(visualized_pairs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained WiFlow pose model.")
    parser.add_argument("--dataset-root", required=True, help="Path to the MM-Fi HDF5 dataset or its parent directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to a WiFlow checkpoint file.")
    parser.add_argument("--output-dir", default="outputs/eval", help="Directory for evaluation visualizations.")
    parser.add_argument("--split-scheme", default=DEFAULT_SPLIT_SCHEME, choices=SPLIT_SCHEMES)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-visualizations", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model, csi_features = load_checkpoint_model(args.checkpoint, device)

    test_dataset = MMFiPoseDataset(
        dataset_root=args.dataset_root,
        split="test",
        split_scheme=args.split_scheme,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    metrics = evaluate_model(model, test_loader, device, csi_features)
    print("--- Test Metrics ---")
    for name in sorted(metrics):
        print(f"{name}: {metrics[name]:.6f}")

    joint_rows, action_rows, environment_rows = collect_metric_breakdowns(
        model,
        test_loader,
        device,
        csi_features,
    )
    output_dir = Path(args.output_dir)
    write_csv_rows(output_dir / "per_joint_metrics.csv", joint_rows)
    write_csv_rows(output_dir / "per_action_metrics.csv", action_rows)
    write_csv_rows(output_dir / "per_environment_metrics.csv", environment_rows)

    saved_count = save_visualizations(
        model,
        test_dataset,
        output_dir,
        device,
        csi_features,
        max_visualizations=args.max_visualizations,
    )
    print(f"Saved visualizations: {saved_count}")


if __name__ == "__main__":
    main()
