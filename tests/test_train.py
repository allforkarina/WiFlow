from __future__ import annotations

import torch

from train import (
    COCO_BONE_EDGES,
    JOINT_LOSS_WEIGHTS,
    LIMB_VECTOR_LOSS_WEIGHTS,
    TrainConfig,
    bone_length_loss,
    compute_lambda_bone,
    compute_losses,
    compute_metrics,
    joint_weighted_pose_loss,
    joint_weighted_scale_normalized_pose_loss,
    limb_vector_loss,
    prepare_model_input,
    scale_normalized_pose_loss,
    weighted_limb_vector_loss,
)


def test_prepare_model_input_preserves_structured_csi_amplitude() -> None:
    batch = {
        "csi_amplitude": torch.randn(2, 3, 114, 10),
        "keypoints": torch.randn(2, 17, 2),
    }

    model_input, target = prepare_model_input(batch, torch.device("cpu"))

    assert model_input.shape == (2, 3, 114, 10)
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

    losses = compute_losses(
        prediction,
        target,
        lambda_bone=0.2,
        beta=0.1,
        scale_norm_loss_weight=0.5,
        limb_vector_loss_weight=0.2,
    )

    expected = (
        losses["pose_loss"]
        + 0.5 * losses["scale_norm_pose_loss"]
        + 0.2 * losses["bone_loss"]
        + 0.2 * losses["limb_vector_loss"]
    )
    assert set(losses) == {
        "loss",
        "pose_loss",
        "scale_norm_pose_loss",
        "bone_loss",
        "limb_vector_loss",
    }
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
    assert config.lr == 2e-5
    assert config.max_lr == 5e-4
    assert config.weight_decay == 5e-4
    assert config.smooth_l1_beta == 0.1
    assert config.grad_clip_norm == 1.0
    assert config.bone_loss_warmup_epochs == 10
    assert config.bone_loss_final_lambda == 0.5
    assert config.scale_norm_loss_weight == 0.5
    assert config.limb_vector_loss_weight == 0.2


def test_scale_normalized_pose_loss_is_zero_for_matching_predictions() -> None:
    target = torch.randn(2, 17, 2)

    loss = scale_normalized_pose_loss(target, target)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_joint_weighted_pose_loss_is_zero_for_matching_predictions() -> None:
    target = torch.randn(2, 17, 2)

    loss = joint_weighted_pose_loss(target, target)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_joint_weighted_scale_normalized_pose_loss_is_zero_for_matching_predictions() -> None:
    target = torch.randn(2, 17, 2)

    loss = joint_weighted_scale_normalized_pose_loss(target, target)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_limb_vector_loss_is_zero_for_matching_skeletons() -> None:
    target = torch.randn(2, 17, 2)

    loss = limb_vector_loss(target, target)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_weighted_limb_vector_loss_is_zero_for_matching_skeletons() -> None:
    target = torch.randn(2, 17, 2)

    loss = weighted_limb_vector_loss(target, target)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_compute_lambda_bone_warms_up_then_reaches_final_value() -> None:
    config = TrainConfig(dataset_root="data/mmfi_pose.h5", epochs=50)

    assert compute_lambda_bone(1, config) == 0.0
    assert compute_lambda_bone(10, config) == 0.0
    assert compute_lambda_bone(11, config) > 0.0
    assert compute_lambda_bone(25, config) > compute_lambda_bone(11, config)
    assert compute_lambda_bone(50, config) == config.bone_loss_final_lambda


def test_joint_and_limb_weight_constants_emphasize_upper_limbs() -> None:
    assert JOINT_LOSS_WEIGHTS[9] > JOINT_LOSS_WEIGHTS[7] > JOINT_LOSS_WEIGHTS[0]
    assert JOINT_LOSS_WEIGHTS[10] > JOINT_LOSS_WEIGHTS[8] > JOINT_LOSS_WEIGHTS[0]
    assert LIMB_VECTOR_LOSS_WEIGHTS[4] > LIMB_VECTOR_LOSS_WEIGHTS[3] > LIMB_VECTOR_LOSS_WEIGHTS[0]
    assert LIMB_VECTOR_LOSS_WEIGHTS[6] > LIMB_VECTOR_LOSS_WEIGHTS[5] > LIMB_VECTOR_LOSS_WEIGHTS[0]
