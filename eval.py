from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from torch.utils.data import DataLoader

from dataloader import MMFiPoseDataset, denormalize_keypoints
from models import WiFlowModel
from train import COCO_BONE_EDGES, compute_metrics, prepare_model_input, select_device


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device) -> WiFlowModel:
    """Load a trained WiFlow model from a checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device)   # load the checkpoint
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing model_state_dict: {checkpoint_path}")

    model = WiFlowModel().to(device)                                # load the model
    model.load_state_dict(checkpoint["model_state_dict"])           # load the model weights
    model.eval()                                                    # eval mode
    return model


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


def evaluate_model(
    model: WiFlowModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model on one dataloader and return averaged metrics."""

    totals: Dict[str, float] = {}
    sample_count = 0

    with torch.no_grad():
        for batch in loader:
            model_input, target = prepare_model_input(batch, device)    # get test data
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
        csi_amplitude = torch.as_tensor(sample["csi_amplitude"], dtype=torch.float32, device=device).unsqueeze(0)
        model_input = csi_amplitude.reshape(1, 342, 10)
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
        axes[0].imshow(model_input.detach().cpu().numpy()[0], aspect="auto", cmap="jet")
        axes[0].set_title(f"CSI Amplitude Heatmap ({action} in {environment})")
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-visualizations", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model = load_checkpoint_model(args.checkpoint, device)

    test_dataset = MMFiPoseDataset(dataset_root=args.dataset_root, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    metrics = evaluate_model(model, test_loader, device)
    print("--- Test Metrics ---")
    for name in sorted(metrics):
        print(f"{name}: {metrics[name]:.6f}")

    saved_count = save_visualizations(
        model,
        test_dataset,
        Path(args.output_dir),
        device,
        max_visualizations=args.max_visualizations,
    )
    print(f"Saved visualizations: {saved_count}")


if __name__ == "__main__":
    main()
