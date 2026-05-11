from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from torch.utils.data import DataLoader, Subset

from dataloader import DEFAULT_SPLIT_SCHEME, SPLIT_SCHEMES, create_data_loaders
from models import AXIAL_ENCODER_MODES, COCO_BONE_EDGES, DECODER_TYPES, WiFlowModel
from pose_targets import build_pcm_paf_targets


DEFAULT_CSI_FEATURES: tuple[str, ...] = ("csi_amplitude", "csi_phase_cos")
SUPPORTED_CSI_FEATURES: tuple[str, ...] = ("csi_amplitude", "csi_phase_cos")
PCK_THRESHOLDS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
RIGHT_SHOULDER_INDEX = 6
LEFT_HIP_INDEX = 11


@dataclass(frozen=True)
class TrainConfig:
    dataset_root: str
    output_dir: str = "outputs/train"
    split_scheme: str = DEFAULT_SPLIT_SCHEME
    csi_features: tuple[str, ...] = DEFAULT_CSI_FEATURES
    axial_mode: str = "spatial_then_temporal"
    decoder_type: str = "joint"
    sequence_length: int = 1
    epochs: int = 50
    batch_size: int = 64
    lr: float = 2e-5
    max_lr: float = 5e-4
    weight_decay: float = 5e-4
    grad_clip_norm: float = 1.0
    bone_loss_weight: float = 0.5
    heatmap_size: int = 36
    heatmap_sigma: float = 1.5
    paf_width: float = 1.0
    paf_loss_weight: float = 1.0
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42
    subset_size: int | None = None


def parse_csi_features(value: str | Iterable[str]) -> tuple[str, ...]:
    """Parse and validate the configured CSI feature names outside the batch hot path."""

    if isinstance(value, str):
        features = tuple(feature.strip() for feature in value.split(",") if feature.strip())
    else:
        features = tuple(value)
    if not features:
        raise ValueError("At least one CSI feature must be selected")
    if len(set(features)) != len(features):
        raise ValueError(f"CSI features must not contain duplicates: {features}")
    unsupported = [feature for feature in features if feature not in SUPPORTED_CSI_FEATURES]
    if unsupported:
        raise ValueError(f"Unsupported CSI features {unsupported}; supported features are {SUPPORTED_CSI_FEATURES}")
    return features


def csi_feature_string(features: Iterable[str]) -> str:
    return ",".join(features)


def prepare_model_input(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
    csi_features: tuple[str, ...] = DEFAULT_CSI_FEATURES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one dataloader batch to model input and labels."""

    feature_tensors = [
        torch.as_tensor(batch[feature_name], dtype=torch.float32, device=device)
        for feature_name in csi_features
    ]
    feature_ndim = feature_tensors[0].ndim
    if feature_ndim not in (4, 5):
        raise ValueError(f"Expected CSI feature tensors with 4 or 5 dimensions, got {feature_ndim}")
    model_input = torch.cat(feature_tensors, dim=2 if feature_ndim == 5 else 1)
    keypoints = torch.as_tensor(batch["keypoints"], dtype=torch.float32, device=device)
    return model_input, keypoints


def effective_sequence_length(split_scheme: str, sequence_length: int) -> int:
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    if split_scheme == "frame_random":
        return 1
    return sequence_length


def bone_length_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    edges: tuple[tuple[int, int], ...] = COCO_BONE_EDGES,
) -> torch.Tensor:
    """Compute L1 loss between predicted and target bone lengths."""

    edge_index = torch.as_tensor(edges, dtype=torch.long, device=prediction.device)
    pred_lengths = torch.linalg.vector_norm(
        prediction[:, edge_index[:, 0]] - prediction[:, edge_index[:, 1]],
        dim=-1,
    )
    target_lengths = torch.linalg.vector_norm(
        target[:, edge_index[:, 0]] - target[:, edge_index[:, 1]],
        dim=-1,
    )
    return F.l1_loss(pred_lengths, target_lengths)


def extract_prediction_keypoints(prediction: Any) -> torch.Tensor:
    if isinstance(prediction, Mapping):
        keypoints = prediction.get("keypoints")
        if not isinstance(keypoints, torch.Tensor):
            raise ValueError("Heatmap decoder output must contain tensor keypoints")
        return keypoints
    if not isinstance(prediction, torch.Tensor):
        raise TypeError(f"Unexpected model prediction type: {type(prediction)!r}")
    return prediction


def compute_losses(
    prediction: Any,
    target: torch.Tensor,
    bone_loss_weight: float = 0.5,
    heatmap_size: int = 36,
    heatmap_sigma: float = 1.5,
    paf_width: float = 1.0,
    paf_loss_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Return training losses for coordinate or heatmap decoder outputs."""

    zero = torch.zeros((), dtype=target.dtype, device=target.device)
    if isinstance(prediction, Mapping):
        stages = prediction.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("Heatmap decoder output must contain non-empty stages")
        pcm_gt, paf_gt = build_pcm_paf_targets(
            target,
            heatmap_size=heatmap_size,
            sigma=heatmap_sigma,
            paf_width=paf_width,
        )
        pcm_total = zero
        paf_total = zero
        for stage in stages:
            pcm_total = pcm_total + F.mse_loss(stage["pcm"], pcm_gt)
            paf_total = paf_total + F.mse_loss(stage["paf"], paf_gt)
        total = pcm_total + paf_loss_weight * paf_total
        return {
            "loss": total,
            "coord_loss": zero,
            "bone_loss": zero,
            "pcm_loss": pcm_total,
            "paf_loss": paf_total,
        }

    coord = F.l1_loss(prediction, target)
    bone = bone_length_loss(prediction, target)
    total = coord + bone_loss_weight * bone
    return {
        "loss": total,
        "coord_loss": coord,
        "bone_loss": bone,
        "pcm_loss": zero,
        "paf_loss": zero,
    }


def compute_torso_scale(target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return torso scales derived from the right-shoulder to left-hip distance."""

    return torch.linalg.vector_norm(
        target[:, RIGHT_SHOULDER_INDEX] - target[:, LEFT_HIP_INDEX],
        dim=-1,
    ).clamp_min(eps)


def mpjpe(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per-joint Euclidean distance."""

    return torch.linalg.vector_norm(prediction - target, dim=-1).mean()


def pck(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Percentage of correct keypoints normalized by right-shoulder to left-hip distance."""

    errors = torch.linalg.vector_norm(prediction - target, dim=-1)
    scale = compute_torso_scale(target, eps=eps)
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
    scheduler: LRScheduler | None = None,
) -> Dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    totals: Dict[str, float] = {}
    sample_count = 0

    for batch in loader:
        model_input, target = prepare_model_input(batch, device, criterion_config.csi_features)

        with torch.set_grad_enabled(is_training):
            prediction = model(model_input)
            losses = compute_losses(
                prediction,
                target,
                bone_loss_weight=criterion_config.bone_loss_weight,
                heatmap_size=criterion_config.heatmap_size,
                heatmap_sigma=criterion_config.heatmap_sigma,
                paf_width=criterion_config.paf_width,
                paf_loss_weight=criterion_config.paf_loss_weight,
            )
            keypoint_prediction = extract_prediction_keypoints(prediction)
            metrics = compute_metrics(keypoint_prediction.detach(), target)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=criterion_config.grad_clip_norm,
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        batch_size = target.shape[0]
        sample_count += batch_size
        for name, value in {**losses, **metrics}.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size

    return average_meter_totals(totals, sample_count)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LRScheduler,
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


def append_csv_row(path: Path, row: Mapping[str, float | int | str]) -> None:
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
    effective_length = effective_sequence_length(config.split_scheme, config.sequence_length)
    if effective_length != config.sequence_length:
        print(
            "frame_random split uses single-frame behavior; "
            f"overriding sequence_length={config.sequence_length} to 1"
        )
        config = replace(config, sequence_length=effective_length)

    torch.manual_seed(config.seed)
    device = select_device(config.device)
    output_dir = Path(config.output_dir)
    input_channels = len(config.csi_features) * 3

    loaders = create_data_loaders(
        dataset_root=config.dataset_root,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
        split_scheme=config.split_scheme,
        sequence_length=config.sequence_length,
    )

    train_loader = maybe_subset_loader(loaders["train"], config.subset_size)
    val_loader = maybe_subset_loader(loaders["val"], config.subset_size)
    model = WiFlowModel(
        input_channels=input_channels,
        axial_mode=config.axial_mode,
        sequence_length=config.sequence_length,
        decoder_type=config.decoder_type,
        heatmap_size=config.heatmap_size,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.max_lr,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=config.max_lr / max(config.lr, 1e-8),
        final_div_factor=1000.0,
    )

    first_batch = next(iter(train_loader))
    model_input, target = prepare_model_input(first_batch, device, config.csi_features)
    with torch.no_grad():
        output = model(model_input)
    keypoint_output = extract_prediction_keypoints(output)
    print(
        "Sanity shapes: "
        f"input={tuple(model_input.shape)}, output={tuple(keypoint_output.shape)}, label={tuple(target.shape)}"
    )
    if keypoint_output.shape != target.shape:
        raise ValueError(
            f"Model output shape {tuple(keypoint_output.shape)} does not match label shape {tuple(target.shape)}"
        )

    best_val_mpjpe = float("inf")
    best_val_pck_0_2 = -float("inf")
    feature_names = csi_feature_string(config.csi_features)
    log_path = output_dir / "train_log.csv"
    for epoch in range(1, config.epochs + 1):
        start_time = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            config,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        val_metrics = run_epoch(model, val_loader, config, device)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - start_time

        row: Dict[str, float | int | str] = {
            "epoch": epoch,
            "csi_features": feature_names,
            "axial_mode": config.axial_mode,
            "decoder_type": config.decoder_type,
            "sequence_length": config.sequence_length,
            "train_loss": train_metrics["loss"],
            "train_coord_loss": train_metrics["coord_loss"],
            "train_bone_loss": train_metrics["bone_loss"],
            "train_pcm_loss": train_metrics["pcm_loss"],
            "train_paf_loss": train_metrics["paf_loss"],
            "train_mpjpe": train_metrics["mpjpe"],
            "train_pck_0_2": train_metrics["pck_0_2"],
            "val_loss": val_metrics["loss"],
            "val_coord_loss": val_metrics["coord_loss"],
            "val_bone_loss": val_metrics["bone_loss"],
            "val_pcm_loss": val_metrics["pcm_loss"],
            "val_paf_loss": val_metrics["paf_loss"],
            "val_mpjpe": val_metrics["mpjpe"],
            "val_pck_0_2": val_metrics["pck_0_2"],
            "val_pck_0_5": val_metrics["pck_0_5"],
            "heatmap_size": config.heatmap_size,
            "heatmap_sigma": config.heatmap_sigma,
            "paf_width": config.paf_width,
            "paf_loss_weight": config.paf_loss_weight,
            "current_lr": current_lr,
            "epoch_time": epoch_time,
        }
        append_csv_row(log_path, row)

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
    parser.add_argument("--split-scheme", default=DEFAULT_SPLIT_SCHEME, choices=SPLIT_SCHEMES)
    parser.add_argument("--csi-features", default=csi_feature_string(DEFAULT_CSI_FEATURES))
    parser.add_argument("--axial-mode", default="spatial_then_temporal", choices=AXIAL_ENCODER_MODES)
    parser.add_argument("--decoder-type", default="joint", choices=DECODER_TYPES)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--bone-loss-weight", type=float, default=0.5)
    parser.add_argument("--heatmap-size", type=int, default=36)
    parser.add_argument("--heatmap-sigma", type=float, default=1.5)
    parser.add_argument("--paf-width", type=float, default=1.0)
    parser.add_argument("--paf-loss-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-size", type=int, default=None)
    args = parser.parse_args()
    args.csi_features = parse_csi_features(args.csi_features)
    return args


def main() -> None:
    args = parse_args()
    config = TrainConfig(**vars(args))
    run_training(config)


if __name__ == "__main__":
    main()
