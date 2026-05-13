from __future__ import annotations

import torch

from models import DECODER_TYPES
from train import (
    TrainConfig,
    bone_length_loss,
    compute_losses,
    compute_metrics,
    parse_args,
    prepare_model_input,
)


def test_prepare_model_input_extracts_csi_and_keypoints() -> None:
    batch = {
        "csi_amplitude": torch.randn(2, 3, 114, 64),
        "keypoints": torch.randn(2, 18, 2),
    }

    model_input, target = prepare_model_input(batch, torch.device("cpu"))

    assert model_input.shape == (2, 3, 114, 64)
    assert target.shape == (2, 18, 2)


def test_bone_length_loss_is_zero_for_matching_skeletons() -> None:
    target = torch.randn(3, 18, 2)

    loss = bone_length_loss(target, target)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_compute_losses_returns_weighted_total() -> None:
    prediction = torch.zeros(2, 18, 2)
    target = torch.ones(2, 18, 2)

    losses = compute_losses(prediction, target, bone_loss_weight=0.5)

    expected = losses["coord_loss"] + 0.5 * losses["bone_loss"]
    assert set(losses) == {"loss", "coord_loss", "bone_loss", "pcm_loss", "paf_loss"}
    assert torch.isclose(losses["loss"], expected)
    assert torch.isclose(losses["pcm_loss"], torch.tensor(0.0))
    assert torch.isclose(losses["paf_loss"], torch.tensor(0.0))


def test_compute_metrics_returns_mpjpe_and_pck_values() -> None:
    target = torch.zeros(1, 18, 2)
    target[0, 6] = torch.tensor([1.0, 0.0])
    target[0, 11] = torch.tensor([0.0, 0.0])
    prediction = target.clone()

    metrics = compute_metrics(prediction, target)

    assert torch.isclose(metrics["mpjpe"], torch.tensor(0.0))
    assert torch.isclose(metrics["pck_0_1"], torch.tensor(1.0))
    assert torch.isclose(metrics["pck_0_5"], torch.tensor(1.0))


def test_train_config_uses_defaults() -> None:
    config = TrainConfig(dataset_root="data/mmfi_pose")

    assert config.epochs == 50
    assert config.batch_size == 64
    assert config.axial_mode == "spatial_then_temporal"
    assert config.decoder_type == "joint"
    assert config.lr == 2e-5
    assert config.max_lr == 5e-4
    assert config.weight_decay == 5e-4
    assert config.grad_clip_norm == 1.0
    assert config.bone_loss_weight == 0.5
    assert config.heatmap_size == 36
    assert config.heatmap_sigma == 1.5
    assert config.paf_width == 1.0
    assert config.paf_loss_weight == 1.0


def test_compute_losses_supports_heatmap_msfn_output() -> None:
    target = torch.full((2, 18, 2), 0.5)
    prediction = {
        "keypoints": target.clone(),
        "stages": [
            {
                "pcm": torch.zeros(2, 18, 36, 36),
                "paf": torch.zeros(2, 38, 36, 36),
            },
            {
                "pcm": torch.zeros(2, 18, 36, 36),
                "paf": torch.zeros(2, 38, 36, 36),
            },
        ],
    }

    losses = compute_losses(prediction, target)

    assert set(losses) == {"loss", "coord_loss", "bone_loss", "pcm_loss", "paf_loss"}
    assert losses["pcm_loss"] > 0
    assert torch.isclose(losses["coord_loss"], torch.tensor(0.0))
    assert torch.isclose(losses["bone_loss"], torch.tensor(0.0))


def test_parse_args_accepts_axial_mode_and_decoder_type(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--dataset-root",
            "data/mmfi_pose",
            "--axial-mode",
            "parallel_concat",
            "--decoder-type",
            "hierarchical",
            "--heatmap-size",
            "40",
        ],
    )

    args = parse_args()

    assert args.axial_mode == "parallel_concat"
    assert args.decoder_type == "hierarchical"
    assert args.heatmap_size == 40
    assert DECODER_TYPES == ("joint", "hierarchical", "heatmap_msfn")