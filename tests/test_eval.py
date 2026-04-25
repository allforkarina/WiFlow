from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import torch

from eval import average_metrics, plot_skeleton, safe_stem, update_metric_totals
from train import COCO_BONE_EDGES

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
