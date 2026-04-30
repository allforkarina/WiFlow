from __future__ import annotations

import torch
from torch import nn

from .wiflow_attention_pooler import WiFlowAttentionPooler
from .wiflow_axial_encoder import WiFlowAxialEncoder
from .wiflow_skeleton_decoder import WiFlowSkeletonDecoder
from .wiflow_spatial_encoder import WiFlowSpatialEncoder


class WiFlowModel(nn.Module):
    """End-to-end WiFlow model that maps CSI features to COCO17 coordinates."""

    def __init__(self, input_channels: int = 6, axial_mode: str = "spatial_then_temporal") -> None:
        super().__init__()
        self.input_channels = input_channels
        self.axial_mode = axial_mode
        self.spatial_encoder = WiFlowSpatialEncoder(input_channels=input_channels)
        self.axial_encoder = WiFlowAxialEncoder(mode=axial_mode)
        self.pooler = WiFlowAttentionPooler()
        self.decoder = WiFlowSkeletonDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_encoder(x)
        x = self.axial_encoder(x)
        x = self.pooler(x)
        return self.decoder(x)
