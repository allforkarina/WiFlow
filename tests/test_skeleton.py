from __future__ import annotations

import torch

from models import OPENPOSE_BONE_EDGES, NUM_OPENPOSE_KEYPOINTS, build_normalized_adjacency


def test_openpose_bone_edges_define_nineteen_valid_edges() -> None:
    assert len(OPENPOSE_BONE_EDGES) == 19
    assert all(0 <= start < NUM_OPENPOSE_KEYPOINTS and 0 <= end < NUM_OPENPOSE_KEYPOINTS for start, end in OPENPOSE_BONE_EDGES)


def test_normalized_adjacency_is_symmetric_with_self_loops() -> None:
    adjacency = build_normalized_adjacency()

    assert adjacency.shape == (18, 18)
    assert torch.allclose(adjacency, adjacency.T)
    assert torch.all(adjacency.diag() > 0)