from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import torch

from eval import (
    average_metrics,
    build_group_metric_rows,
    build_joint_metric_rows,
    compute_joint_errors,
    compute_joint_pck,
    load_checkpoint_model,
    plot_skeleton,
    safe_stem,
    update_group_metric_totals,
    update_metric_totals,
    write_csv_rows,
)
from models import COCO_BONE_EDGES, WiFlowModel

matplotlib.use("Agg")


def test_update_metric_totals_weights_by_batch_size() -> None:
    totals: dict[str, float] = {}

    update_metric_totals(totals, {"mpjpe": torch.tensor(2.0)}, batch_size=4)
    update_metric_totals(totals, {"mpjpe": torch.tensor(6.0)}, batch_size=2)

    averaged = average_metrics(totals, sample_count=6)

    assert averaged["mpjpe"] == (2.0 * 4 + 6.0 * 2) / 6


def test_safe_stem_replaces_path_unsafe_characters() -> None:
    stem = safe_stem("A01", "env/1", "frame:001")

    assert stem == "A01_env_1_frame_001"


def test_plot_skeleton_runs_on_agg_backend() -> None:
    keypoints = torch.zeros(17, 2).numpy()
    fig, ax = plt.subplots()

    plot_skeleton(ax, keypoints, COCO_BONE_EDGES, "Skeleton", color="blue")

    assert ax.get_title() == "Skeleton"
    plt.close(fig)


def test_compute_joint_errors_and_pck_preserve_joint_shape() -> None:
    target = torch.zeros(2, 17, 2)
    target[:, 6] = torch.tensor([1.0, 0.0])
    prediction = target.clone()

    joint_errors = compute_joint_errors(prediction, target)
    joint_pck = compute_joint_pck(prediction, target)

    assert joint_errors.shape == (2, 17)
    assert joint_pck.shape == (2, 17)
    assert torch.allclose(joint_errors, torch.zeros_like(joint_errors))
    assert torch.allclose(joint_pck, torch.ones_like(joint_pck))


def test_group_metric_rows_include_expected_fields() -> None:
    joint_errors = torch.tensor([[1.0] * 17, [3.0] * 17])
    joint_pck = torch.tensor([[1.0] * 17, [0.0] * 17])
    totals: dict[str, dict[str, float]] = {}

    update_group_metric_totals(totals, ["A01", "A01"], joint_errors, joint_pck)
    rows = build_group_metric_rows(totals, "action")

    assert rows == [
        {
            "action": "A01",
            "sample_count": 2,
            "mpjpe": 2.0,
            "pck_0_2": 0.5,
        }
    ]


def test_build_joint_metric_rows_aggregates_batches() -> None:
    joint_errors = [torch.tensor([[1.0] * 17]), torch.tensor([[3.0] * 17])]
    joint_pck = [torch.tensor([[1.0] * 17]), torch.tensor([[0.0] * 17])]

    rows = build_joint_metric_rows(joint_errors, joint_pck)

    assert len(rows) == 17
    assert rows[0]["joint_index"] == 0
    assert rows[0]["sample_count"] == 2
    assert rows[0]["mpjpe"] == 2.0
    assert rows[0]["pck_0_2"] == 0.5


def test_write_csv_rows_writes_header_and_rows(tmp_path) -> None:
    path = tmp_path / "metrics.csv"

    write_csv_rows(path, [{"joint_index": 0, "sample_count": 2, "mpjpe": 1.0, "pck_0_2": 0.5}])

    contents = path.read_text(encoding="utf-8")
    assert "joint_index,sample_count,mpjpe,pck_0_2" in contents
    assert "0,2,1.0,0.5" in contents


def test_load_checkpoint_model_uses_train_config_csi_features(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pth"
    model = WiFlowModel(input_channels=6)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_config": {"csi_features": ("csi_amplitude", "csi_phase_cos")},
        },
        checkpoint_path,
    )

    loaded_model, csi_features = load_checkpoint_model(checkpoint_path, torch.device("cpu"))

    assert isinstance(loaded_model, WiFlowModel)
    assert loaded_model.input_channels == 6
    assert csi_features == ("csi_amplitude", "csi_phase_cos")


def test_load_checkpoint_model_requires_csi_features(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": WiFlowModel().state_dict(), "train_config": {}}, checkpoint_path)

    try:
        load_checkpoint_model(checkpoint_path, torch.device("cpu"))
    except KeyError as exc:
        assert "csi_features" in str(exc)
    else:
        raise AssertionError("Expected checkpoint loading to require train_config.csi_features")
