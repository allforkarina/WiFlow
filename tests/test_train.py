from __future__ import annotations

import torch

from train import (
    COCO_BONE_EDGES,
    TrainConfig,
    bone_length_loss,
    compute_losses,
    compute_metrics,
    prepare_model_input,
)


def test_prepare_model_input_flattens_csi_amplitude() -> None:
    batch = {
        "csi_amplitude": torch.randn(2, 3, 114, 10),
        "keypoints": torch.randn(2, 17, 2),
    }

    model_input, target = prepare_model_input(batch, torch.device("cpu"))

    assert model_input.shape == (2, 342, 10)
    assert target.shape == (2, 17, 2)


def test_coco_bone_edges_define_fourteen_valid_edges() -> None:
    assert len(COCO_BONE_EDGES) == 14
    assert all(0 <= start < 17 and 0 <= end < 17 for start, end in COCO_BONE_EDGES)


def test_bone_length_loss_is_zero_for_matching_skeletons() -> None:
    target = torch.randn(3, 17, 2)

    loss = bone_length_loss(target, target)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_compute_losses_returns_weighted_total() -> None:
    prediction = torch.zeros(2, 17, 2)
    target = torch.ones(2, 17, 2)

    losses = compute_losses(prediction, target, lambda_bone=0.2, beta=0.1)

    expected = losses["pose_loss"] + 0.2 * losses["bone_loss"]
    assert set(losses) == {"loss", "pose_loss", "bone_loss"}
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


def test_train_config_uses_paper_defaults() -> None:
    config = TrainConfig(dataset_root="data/mmfi_pose.h5")

    assert config.epochs == 50
    assert config.batch_size == 64
    assert config.lr == 1e-4
    assert config.weight_decay == 5e-5
    assert config.lambda_bone == 0.2
    assert config.smooth_l1_beta == 0.1
