from __future__ import annotations

import torch


NUM_COCO_KEYPOINTS = 17
COCO_BONE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
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


def build_normalized_adjacency(
    num_nodes: int = NUM_COCO_KEYPOINTS,
    edges: tuple[tuple[int, int], ...] = COCO_BONE_EDGES,
) -> torch.Tensor:
    """Build symmetric normalized adjacency with self-loops for COCO keypoints."""

    adjacency = torch.eye(num_nodes, dtype=torch.float32)
    for start, end in edges:
        adjacency[start, end] = 1.0
        adjacency[end, start] = 1.0

    degree = adjacency.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    return degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]
