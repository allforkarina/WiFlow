from __future__ import annotations

import torch

from dataloader import DEFAULT_SPLIT_SCHEME
from train import (
    DEFAULT_CSI_FEATURES,
    SUPPORTED_CSI_FEATURES,
    TrainConfig,
    bone_length_loss,
    compute_losses,
    compute_metrics,
    csi_feature_string,
    parse_args,
    parse_csi_features,
    prepare_model_input,
)


def test_parse_csi_features_supports_default_and_amp_only() -> None:
    assert parse_csi_features("csi_amplitude,csi_phase_cos") == DEFAULT_CSI_FEATURES
    assert parse_csi_features("csi_amplitude") == ("csi_amplitude",)
    assert SUPPORTED_CSI_FEATURES == ("csi_amplitude", "csi_phase_cos")


def test_parse_csi_features_rejects_empty_duplicate_and_unknown() -> None:
    for value in ("", "csi_amplitude,csi_amplitude", "csi_phase_sin"):
        try:
            parse_csi_features(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected parse_csi_features to reject {value!r}")


def test_csi_feature_string_joins_feature_names() -> None:
    assert csi_feature_string(DEFAULT_CSI_FEATURES) == "csi_amplitude,csi_phase_cos"


def test_prepare_model_input_concatenates_configured_csi_features() -> None:
    amplitude = torch.ones(2, 3, 114, 10)
    phase_cos = torch.zeros(2, 3, 114, 10)
    batch = {
        "csi_amplitude": amplitude,
        "csi_phase_cos": phase_cos,
        "keypoints": torch.randn(2, 17, 2),
    }

    model_input, target = prepare_model_input(batch, torch.device("cpu"), DEFAULT_CSI_FEATURES)

    assert model_input.shape == (2, 6, 114, 10)
    assert target.shape == (2, 17, 2)
    assert torch.equal(model_input[:, :3], amplitude)
    assert torch.equal(model_input[:, 3:], phase_cos)


def test_prepare_model_input_supports_amp_only() -> None:
    batch = {
        "csi_amplitude": torch.randn(2, 3, 114, 10),
        "keypoints": torch.randn(2, 17, 2),
    }

    model_input, target = prepare_model_input(batch, torch.device("cpu"), ("csi_amplitude",))

    assert model_input.shape == (2, 3, 114, 10)
    assert target.shape == (2, 17, 2)


def test_bone_length_loss_is_zero_for_matching_skeletons() -> None:
    target = torch.randn(3, 17, 2)

    loss = bone_length_loss(target, target)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_compute_losses_returns_weighted_total() -> None:
    prediction = torch.zeros(2, 17, 2)
    target = torch.ones(2, 17, 2)

    losses = compute_losses(prediction, target, bone_loss_weight=0.5)

    expected = losses["coord_loss"] + 0.5 * losses["bone_loss"]
    assert set(losses) == {"loss", "coord_loss", "bone_loss"}
    assert torch.isclose(losses["loss"], expected)


def test_compute_metrics_returns_mpjpe_and_pck_values() -> None:
    target = torch.zeros(1, 17, 2)
    target[0, 6] = torch.tensor([1.0, 0.0])
    target[0, 11] = torch.tensor([0.0, 0.0])
    prediction = target.clone()

    metrics = compute_metrics(prediction, target)

    assert torch.isclose(metrics["mpjpe"], torch.tensor(0.0))
    assert torch.isclose(metrics["pck_0_1"], torch.tensor(1.0))
    assert torch.isclose(metrics["pck_0_5"], torch.tensor(1.0))


def test_train_config_uses_refactor_defaults() -> None:
    config = TrainConfig(dataset_root="data/mmfi_pose.h5")

    assert config.epochs == 50
    assert config.batch_size == 64
    assert config.split_scheme == DEFAULT_SPLIT_SCHEME
    assert config.csi_features == DEFAULT_CSI_FEATURES
    assert config.axial_mode == "spatial_then_temporal"
    assert config.lr == 2e-5
    assert config.max_lr == 5e-4
    assert config.weight_decay == 5e-4
    assert config.grad_clip_norm == 1.0
    assert config.bone_loss_weight == 0.5


def test_parse_args_accepts_axial_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--dataset-root",
            "data/mmfi_pose.h5",
            "--axial-mode",
            "parallel_concat",
        ],
    )

    args = parse_args()

    assert args.axial_mode == "parallel_concat"
