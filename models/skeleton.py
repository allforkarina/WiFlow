"""H36M-17 skeleton topology constants and adjacency builder."""

from __future__ import annotations

import torch


NUM_H36M_KEYPOINTS = 17

H36M17_NAMES = [
    "pelvis",
    "r_hip",
    "r_knee",
    "r_ankle",
    "l_hip",
    "l_knee",
    "l_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
]

H36M_BONE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3),       # right leg
    (0, 4), (4, 5), (5, 6),       # left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # spine -> head
    (8, 11), (11, 12), (12, 13),  # left arm
    (8, 14), (14, 15), (15, 16),  # right arm
)


def build_normalized_adjacency(
    num_nodes: int = NUM_H36M_KEYPOINTS,
    edges: tuple[tuple[int, int], ...] = H36M_BONE_EDGES,
) -> torch.Tensor:
    """Build symmetric normalized adjacency with self-loops for H36M-17 keypoints."""

    adjacency = torch.eye(num_nodes, dtype=torch.float32)
    for start, end in edges:
        adjacency[start, end] = 1.0
        adjacency[end, start] = 1.0

    degree = adjacency.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    return degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]
