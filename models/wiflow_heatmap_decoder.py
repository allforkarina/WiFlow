from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .skeleton import H36M_BONE_EDGES, NUM_H36M_KEYPOINTS

_H36M_PAF_CHANNELS = 2 * len(H36M_BONE_EDGES)


class WiFlowHeatmapDecoder(nn.Module):
    """One MSFN stage that predicts H36M-17 PCM and PAF heatmaps."""

    def __init__(
        self,
        feature_channels: int = 128,
        hidden_channels: int = 512,
        pcm_channels: int = NUM_H36M_KEYPOINTS,
        paf_channels: int = _H36M_PAF_CHANNELS,
    ) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.SiLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
        )
        self.pcm_head = nn.Conv2d(hidden_channels, pcm_channels, kernel_size=1)
        self.paf_head = nn.Conv2d(hidden_channels, paf_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(x)
        x = self.bottleneck(x)
        return self.pcm_head(x), self.paf_head(x)


class WiFlowPAPM(nn.Module):
    """Pose-aware feature modulation using previous-stage PCM/PAF heatmaps."""

    def __init__(self, feature_channels: int = 128, heatmap_channels: int = NUM_H36M_KEYPOINTS + _H36M_PAF_CHANNELS) -> None:
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.Linear(heatmap_channels * 2, feature_channels),
            nn.SiLU(),
            nn.Linear(feature_channels, feature_channels),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(heatmap_channels, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, features: torch.Tensor, heatmaps: torch.Tensor) -> torch.Tensor:
        avg_pool = heatmaps.mean(dim=(2, 3))
        max_pool = heatmaps.amax(dim=(2, 3))
        channel_gate = self.channel_gate(torch.cat((avg_pool, max_pool), dim=1))[:, :, None, None]
        spatial_gate = self.spatial_gate(heatmaps)
        return self.refine(features * (1.0 + channel_gate) * (1.0 + spatial_gate))


class WiFlowMSFNDecoder(nn.Module):
    """Multi-stage PCM/PAF decoder aligned with MultiFormer-style MSFN."""

    def __init__(
        self,
        input_channels: int = 256,
        feature_channels: int = 128,
        hidden_channels: int = 512,
        stages: int = 3,
        heatmap_size: int = 36,
        pcm_channels: int = NUM_H36M_KEYPOINTS,
        paf_channels: int = _H36M_PAF_CHANNELS,
    ) -> None:
        super().__init__()
        if stages < 1:
            raise ValueError("stages must be at least 1")
        self.input_channels = input_channels
        self.feature_channels = feature_channels
        self.hidden_channels = hidden_channels
        self.stages = stages
        self.heatmap_size = heatmap_size
        self.pcm_channels = pcm_channels
        self.paf_channels = paf_channels
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.SiLU(inplace=True),
        )
        self.decoders = nn.ModuleList(
            WiFlowHeatmapDecoder(
                feature_channels=feature_channels,
                hidden_channels=hidden_channels,
                pcm_channels=pcm_channels,
                paf_channels=paf_channels,
            )
            for _ in range(stages)
        )
        self.papms = nn.ModuleList(
            WiFlowPAPM(feature_channels=feature_channels, heatmap_channels=pcm_channels + paf_channels)
            for _ in range(stages - 1)
        )

    def forward(self, x: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        if x.ndim != 4:
            raise ValueError("WiFlowMSFNDecoder expects input shaped [B, 256, H, W]")
        current = self.input_projection(x)
        current = F.interpolate(
            current,
            size=(self.heatmap_size, self.heatmap_size),
            mode="bilinear",
            align_corners=False,
        )
        outputs: list[dict[str, torch.Tensor]] = []
        for index, decoder in enumerate(self.decoders):
            pcm, paf = decoder(current)
            outputs.append({"pcm": pcm, "paf": paf})
            if index < len(self.papms):
                current = self.papms[index](current, torch.cat((pcm, paf), dim=1))
        return outputs
