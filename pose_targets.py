from __future__ import annotations

import torch

from models.skeleton import H36M_BONE_EDGES, NUM_H36M_KEYPOINTS


def keypoints_to_heatmap_coords(
    keypoints: torch.Tensor,
    heatmap_size: int,
    pose_range: tuple[float, float] = (-0.8, 0.8),
) -> torch.Tensor:
    """Map H36M-17 keypoints from pose_range to heatmap coordinates."""

    if keypoints.ndim != 3 or keypoints.shape[-2:] != (NUM_H36M_KEYPOINTS, 2):
        raise ValueError(f"Expected keypoints shaped [B, 17, 2], got {tuple(keypoints.shape)}")
    if heatmap_size < 2:
        raise ValueError("heatmap_size must be at least 2")
    pose_min, pose_max = pose_range
    span = pose_max - pose_min
    return ((keypoints - pose_min) / span * float(heatmap_size - 1)).clamp(0.0, float(heatmap_size - 1))


def build_pcm_targets(
    keypoints: torch.Tensor,
    heatmap_size: int = 36,
    sigma: float = 1.5,
    pose_range: tuple[float, float] = (-0.8, 0.8),
) -> torch.Tensor:
    """Build per-joint Gaussian PCM targets from H36M-17 keypoints in pose_range."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    coords = keypoints_to_heatmap_coords(keypoints, heatmap_size, pose_range=pose_range)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(heatmap_size, dtype=keypoints.dtype, device=keypoints.device),
        torch.arange(heatmap_size, dtype=keypoints.dtype, device=keypoints.device),
        indexing="ij",
    )
    dx = grid_x[None, None] - coords[..., 0, None, None]
    dy = grid_y[None, None] - coords[..., 1, None, None]
    dist2 = dx.square() + dy.square()
    return torch.exp(-dist2 / (2.0 * sigma * sigma))


def build_paf_targets(
    keypoints: torch.Tensor,
    heatmap_size: int = 36,
    width: float = 1.0,
    edges: tuple[tuple[int, int], ...] = H36M_BONE_EDGES,
    pose_range: tuple[float, float] = (-0.8, 0.8),
) -> torch.Tensor:
    """Build H36M-17 bone PAF targets from keypoints in pose_range."""

    if width <= 0:
        raise ValueError("width must be positive")
    coords = keypoints_to_heatmap_coords(keypoints, heatmap_size, pose_range=pose_range)
    edge_index = torch.as_tensor(edges, dtype=torch.long, device=keypoints.device)
    p1 = coords[:, edge_index[:, 0]]
    p2 = coords[:, edge_index[:, 1]]
    limb = p2 - p1
    length = torch.linalg.vector_norm(limb, dim=-1).clamp_min(1e-6)
    unit = limb / length[..., None]

    grid_y, grid_x = torch.meshgrid(
        torch.arange(heatmap_size, dtype=keypoints.dtype, device=keypoints.device),
        torch.arange(heatmap_size, dtype=keypoints.dtype, device=keypoints.device),
        indexing="ij",
    )
    rel_x = grid_x[None, None] - p1[..., 0, None, None]
    rel_y = grid_y[None, None] - p1[..., 1, None, None]
    proj = rel_x * unit[..., 0, None, None] + rel_y * unit[..., 1, None, None]
    proj_clamped = proj.clamp_min(0.0).minimum(length[..., None, None])
    closest_x = p1[..., 0, None, None] + proj_clamped * unit[..., 0, None, None]
    closest_y = p1[..., 1, None, None] + proj_clamped * unit[..., 1, None, None]
    dist = torch.sqrt((grid_x[None, None] - closest_x).square() + (grid_y[None, None] - closest_y).square())
    mask = (proj >= 0.0) & (proj <= length[..., None, None]) & (dist <= width)

    paf = torch.zeros(
        keypoints.shape[0],
        len(edges),
        2,
        heatmap_size,
        heatmap_size,
        dtype=keypoints.dtype,
        device=keypoints.device,
    )
    paf[:, :, 0] = torch.where(mask, unit[..., 0, None, None], paf[:, :, 0])
    paf[:, :, 1] = torch.where(mask, unit[..., 1, None, None], paf[:, :, 1])
    return paf.flatten(1, 2)


def build_pcm_paf_targets(
    keypoints: torch.Tensor,
    heatmap_size: int = 36,
    sigma: float = 1.5,
    paf_width: float = 1.0,
    pose_range: tuple[float, float] = (-0.8, 0.8),
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        build_pcm_targets(keypoints, heatmap_size=heatmap_size, sigma=sigma, pose_range=pose_range),
        build_paf_targets(keypoints, heatmap_size=heatmap_size, width=paf_width, pose_range=pose_range),
    )


def decode_pcm_argmax(pcm: torch.Tensor, pose_range: tuple[float, float] = (-0.8, 0.8)) -> torch.Tensor:
    """Decode PCM heatmaps to H36M-17 coordinates in pose_range."""

    if pcm.ndim != 4 or pcm.shape[1] != NUM_H36M_KEYPOINTS:
        raise ValueError(f"Expected PCM shaped [B, 17, H, W], got {tuple(pcm.shape)}")
    _, _, height, width = pcm.shape
    pose_min, pose_max = pose_range
    span = pose_max - pose_min
    flat_indices = pcm.flatten(2).argmax(dim=-1)
    x = (flat_indices % width).to(dtype=pcm.dtype) / float(max(width - 1, 1)) * span + pose_min
    y = torch.div(flat_indices, width, rounding_mode="floor").to(dtype=pcm.dtype) / float(max(height - 1, 1)) * span + pose_min
    return torch.stack((x, y), dim=-1)
