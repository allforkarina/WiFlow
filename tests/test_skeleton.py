from __future__ import annotations

import torch

from models import COCO_BONE_EDGES, NUM_COCO_KEYPOINTS, build_normalized_adjacency


def test_coco_bone_edges_define_sixteen_valid_edges() -> None:
    assert len(COCO_BONE_EDGES) == 16
    assert all(0 <= start < NUM_COCO_KEYPOINTS and 0 <= end < NUM_COCO_KEYPOINTS for start, end in COCO_BONE_EDGES)


def test_normalized_adjacency_is_symmetric_with_self_loops() -> None:
    adjacency = build_normalized_adjacency()

    assert adjacency.shape == (17, 17)
    assert torch.allclose(adjacency, adjacency.T)
    assert torch.all(adjacency.diag() > 0)
