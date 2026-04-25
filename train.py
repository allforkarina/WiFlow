from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from dataloader import create_data_loaders
from models import WiFlowModel


COCO_BONE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 5),
    (0, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
PCK_THRESHOLDS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)   # PCK Threshold
RIGHT_SHOULDER_INDEX = 6                                        # right shoulder keypoint indice
LEFT_HIP_INDEX = 11                                             # left hip keypoint indice


@dataclass(frozen=True)
class TrainConfig:
    dataset_root: str
    output_dir: str = "outputs/train"                           # directory for logs and checkpoints
    epochs: int = 50                                            # training epochs
    batch_size: int = 64                                        # batch size
    lr: float = 1e-4                                            # learning rate
    weight_decay: float = 5e-5                                  # weight decay
    lambda_bone: float = 0.2                                    # bone loss weight
    smooth_l1_beta: float = 0.1                                 # Smooth L1 loss beta
    num_workers: int = 0                                        # number of data loading workers
    device: str = "cuda"                                        # device to use
    seed: int = 42
    subset_size: int | None = None


def prepare_model_input(batch: Mapping[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one dataloader batch to model input [B, 3, 114, 10] and labels [B, 17, 2]."""

    csi_amplitude = torch.as_tensor(batch["csi_amplitude"], dtype=torch.float32, device=device)
    keypoints = torch.as_tensor(batch["keypoints"], dtype=torch.float32, device=device)

    if csi_amplitude.ndim != 4 or csi_amplitude.shape[1:] != (3, 114, 10):
        raise ValueError(f"Expected csi_amplitude shape [B, 3, 114, 10], got {tuple(csi_amplitude.shape)}")
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 2):
        raise ValueError(f"Expected keypoints shape [B, 17, 2], got {tuple(keypoints.shape)}")

    return csi_amplitude, keypoints                                         # [B, 3, 114, 10], [B, 17, 2]


def bone_length_loss(
    prediction: torch.Tensor,                               # [B, 17, 2]
    target: torch.Tensor,                                   # [B, 17, 2]
    edges: Sequence[tuple[int, int]] = COCO_BONE_EDGES,     # [B, E, 2] keypoints connection
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute Smooth L1 loss between predicted and target bone lengths."""

    # turn edges connection list into a tensor [E, 2], from what to what
    edge_index = torch.as_tensor(edges, dtype=torch.long, device=prediction.device)
    
    pred_lengths = torch.linalg.vector_norm(
        # dim = 0 -> batch, so the pred_length of all connection is from [x1, y1] to [x2, y2]
        prediction[:, edge_index[:, 0]] - prediction[:, edge_index[:, 1]],
        dim=-1,
    )
    target_lengths = torch.linalg.vector_norm(
        # dim = 0 -> batch, so the pred_length of all connection is from [x1, y1] to [x2, y2]
        target[:, edge_index[:, 0]] - target[:, edge_index[:, 1]],
        dim=-1,
    )
    return F.smooth_l1_loss(pred_lengths, target_lengths, beta=beta)


def compute_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lambda_bone: float = 0.2,
    beta: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Return total, pose, and bone losses for one batch."""

    pose = F.smooth_l1_loss(prediction, target, beta=beta)                  # pose estimation loss
    bone = bone_length_loss(prediction, target, beta=beta)                  # body constraint loss
    total = pose + lambda_bone * bone
    return {"loss": total, "pose_loss": pose, "bone_loss": bone}


def mpjpe(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per-joint Euclidean distance."""

    return torch.linalg.vector_norm(prediction - target, dim=-1).mean()     # mpjpe metric


def pck(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Percentage of correct keypoints normalized by right-shoulder to left-hip distance."""

    errors = torch.linalg.vector_norm(prediction - target, dim=-1)
    scale = torch.linalg.vector_norm(
        target[:, RIGHT_SHOULDER_INDEX] - target[:, LEFT_HIP_INDEX],
        dim=-1,
    ).clamp_min(eps)
    return (errors < (scale[:, None] * threshold)).float().mean()


def compute_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    metrics = {"mpjpe": mpjpe(prediction, target)}
    for threshold in PCK_THRESHOLDS:
        metrics[f"pck_{threshold:.1f}".replace(".", "_")] = pck(prediction, target, threshold)
    return metrics


def average_meter_totals(totals: Dict[str, float], count: int) -> Dict[str, float]:
    return {name: value / max(count, 1) for name, value in totals.items()}


def run_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    criterion_config: TrainConfig,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> Dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    totals: Dict[str, float] = {}
    sample_count = 0

    for batch in loader:
        model_input, target = prepare_model_input(batch, device)    # load batch data

        with torch.set_grad_enabled(is_training):
            prediction = model(model_input)                         # predict
            losses = compute_losses(                                # loss
                prediction,
                target,
                lambda_bone=criterion_config.lambda_bone,
                beta=criterion_config.smooth_l1_beta,
            )
            metrics = compute_metrics(prediction.detach(), target)  # evaluat the metrics

            if is_training:
                optimizer.zero_grad(set_to_none=True)               # backpropagation
                losses["loss"].backward()
                optimizer.step()

        batch_size = target.shape[0]
        sample_count += batch_size                                  # calculate the total loss
        for name, value in {**losses, **metrics}.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size

    return average_meter_totals(totals, sample_count)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_metric: float,
    config: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "train_config": asdict(config),
        },
        path,
    )


def append_csv_row(path: Path, row: Mapping[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def maybe_subset_loader(loader: DataLoader, subset_size: int | None) -> DataLoader:
    if subset_size is None:
        return loader
    subset_indices = list(range(min(subset_size, len(loader.dataset))))
    return DataLoader(
        Subset(loader.dataset, subset_indices),
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
    )


def select_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def run_training(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    output_dir = Path(config.output_dir)

    loaders = create_data_loaders(              # load data
        dataset_root=config.dataset_root,       # root
        batch_size=config.batch_size,           # batch size
        seed=config.seed,
        num_workers=config.num_workers,         # num_workers for data loading
    )

    train_loader = maybe_subset_loader(loaders["train"], config.subset_size)
    val_loader = maybe_subset_loader(loaders["val"], config.subset_size)    
    model = WiFlowModel().to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
    )

    # dry run, sanity check for model forward pass and output shape
    first_batch = next(iter(train_loader))
    model_input, target = prepare_model_input(first_batch, device)
    with torch.no_grad():
        output = model(model_input)
    print(f"Sanity shapes: input={tuple(model_input.shape)}, output={tuple(output.shape)}, label={tuple(target.shape)}")
    
    if output.shape != target.shape:
        raise ValueError(f"Model output shape {tuple(output.shape)} does not match label shape {tuple(target.shape)}")

    # ======== formal training loop ========

    best_val_mpjpe = float("inf")
    best_val_pck_0_2 = -float("inf")
    log_path = output_dir / "train_log.csv"
    for epoch in range(1, config.epochs + 1):
        start_time = time.perf_counter()                                # epoch start time
        train_metrics = run_epoch(model, train_loader, config, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, config, device)      # validation metrics
        scheduler.step(val_metrics["mpjpe"])                            # calculate mpjpe
        current_lr = optimizer.param_groups[0]["lr"]                    # current learning rate
        epoch_time = time.perf_counter() - start_time                   # epoch duration
        
        row: Dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_pose_loss": train_metrics["pose_loss"],
            "train_bone_loss": train_metrics["bone_loss"],
            "train_mpjpe": train_metrics["mpjpe"],
            "train_pck_0_2": train_metrics["pck_0_2"],
            "val_loss": val_metrics["loss"],
            "val_pose_loss": val_metrics["pose_loss"],
            "val_bone_loss": val_metrics["bone_loss"],
            "val_mpjpe": val_metrics["mpjpe"],
            "val_pck_0_2": val_metrics["pck_0_2"],
            "val_pck_0_5": val_metrics["pck_0_5"],
            "current_lr": current_lr,
            "epoch_time": epoch_time,
        }
        append_csv_row(log_path, row)                                   # add to csv log

        save_checkpoint(
            output_dir / "last.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_metric=val_metrics["mpjpe"],
            config=config,
        )
        
        if val_metrics["mpjpe"] < best_val_mpjpe:
            best_val_mpjpe = val_metrics["mpjpe"]
            save_checkpoint(
                output_dir / "best_val_mpjpe.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric=best_val_mpjpe,
                config=config,
            )
        if val_metrics["pck_0_2"] > best_val_pck_0_2:
            best_val_pck_0_2 = val_metrics["pck_0_2"]
            save_checkpoint(
                output_dir / "best_val_pck_0_2.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric=best_val_pck_0_2,
                config=config,
            )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_mpjpe={val_metrics['mpjpe']:.6f} "
            f"val_pck_0_2={val_metrics['pck_0_2']:.6f} "
            f"lr={current_lr:.2e} "
            f"epoch_time={epoch_time:.1f}s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the WiFlow pose model.")
    parser.add_argument("--dataset-root", required=True, help="Path to the MM-Fi HDF5 dataset or its parent directory.")
    parser.add_argument("--output-dir", default="outputs/train", help="Directory for logs and checkpoints.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--lambda-bone", type=float, default=0.2)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(**vars(args))
    run_training(config)


if __name__ == "__main__":
    main()
