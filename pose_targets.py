from __future__ import annotations

import torch

from models.skeleton import NUM_OPENPOSE_KEYPOINTS, OPENPOSE_BONE_EDGES


def keypoints_to_heatmap_coords(keypoints: torch.Tensor, heatmap_size: int) -> torch.Tensor:
    """Map normalized OpenPose keypoints in [0, 1] to heatmap coordinates."""

    if keypoints.ndim != 3 or keypoints.shape[-2:] != (NUM_OPENPOSE_KEYPOINTS, 2):
        raise ValueError(f"Expected keypoints shaped [B, 18, 2], got {tuple(keypoints.shape)}")
    if heatmap_size < 2:
        raise ValueError("heatmap_size must be at least 2")
    return keypoints.clamp(0.0, 1.0) * float(heatmap_size - 1)


def build_pcm_targets(
    keypoints: torch.Tensor,
    heatmap_size: int = 36,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Build per-joint Gaussian PCM targets from normalized OpenPose18 keypoints."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    coords = keypoints_to_heatmap_coords(keypoints, heatmap_size)
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
    edges: tuple[tuple[int, int], ...] = OPENPOSE_BONE_EDGES,
) -> torch.Tensor:
    """Build OpenPose bone PAF targets with one x/y vector channel pair per edge."""

    if width <= 0:
        raise ValueError("width must be positive")
    coords = keypoints_to_heatmap_coords(keypoints, heatmap_size)
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
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        build_pcm_targets(keypoints, heatmap_size=heatmap_size, sigma=sigma),
        build_paf_targets(keypoints, heatmap_size=heatmap_size, width=paf_width),
    )


def decode_pcm_argmax(pcm: torch.Tensor) -> torch.Tensor:
    """Decode PCM heatmaps to normalized OpenPose18 coordinates with per-channel argmax."""

    if pcm.ndim != 4 or pcm.shape[1] != NUM_OPENPOSE_KEYPOINTS:
        raise ValueError(f"Expected PCM shaped [B, 18, H, W], got {tuple(pcm.shape)}")
    _, _, height, width = pcm.shape
    flat_indices = pcm.flatten(2).argmax(dim=-1)
    x = (flat_indices % width).to(dtype=pcm.dtype) / float(max(width - 1, 1))
    y = torch.div(flat_indices, width, rounding_mode="floor").to(dtype=pcm.dtype) / float(max(height - 1, 1))
    return torch.stack((x, y), dim=-1)
