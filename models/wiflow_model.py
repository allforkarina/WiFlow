from __future__ import annotations

import torch
from torch import nn

from .wiflow_attention_pooler import WiFlowAttentionPooler
from .wiflow_axial_encoder import WiFlowAxialEncoder
from .wiflow_skeleton_decoder import WiFlowSkeletonDecoder
from .wiflow_spatial_encoder import WiFlowSpatialEncoder
from .wiflow_temporal_encoder import WiFlowTemporalEncoder


class WiFlowModel(nn.Module):
    """End-to-end WiFlow model that maps CSI features to COCO17 coordinates."""

    def __init__(
        self,
        input_channels: int = 6,
        axial_mode: str = "spatial_then_temporal",
        sequence_length: int = 1,
    ) -> None:
        super().__init__()
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")
        self.input_channels = input_channels
        self.axial_mode = axial_mode
        self.sequence_length = sequence_length
        self.spatial_encoder = WiFlowSpatialEncoder(input_channels=input_channels)
        self.axial_encoder = WiFlowAxialEncoder(mode=axial_mode)
        self.pooler = WiFlowAttentionPooler()
        self.temporal_encoder = (
            WiFlowTemporalEncoder(sequence_length=sequence_length)
            if sequence_length > 1
            else None
        )
        self.decoder = WiFlowSkeletonDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sequence_length == 1:
            if x.ndim != 4:
                raise ValueError("single-frame WiFlowModel expects input shaped [B, C, 114, 10]")
            x = self.spatial_encoder(x)
            x = self.axial_encoder(x)
            x = self.pooler(x)
            return self.decoder(x)

        if x.ndim != 5:
            raise ValueError("temporal WiFlowModel expects input shaped [B, N, C, 114, 10]")
        if x.shape[1] != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {x.shape[1]}")

        batch_size, sequence_length, channels, subcarriers, frames = x.shape
        x = x.reshape(batch_size * sequence_length, channels, subcarriers, frames)
        x = self.spatial_encoder(x)
        x = self.axial_encoder(x)
        x = self.pooler(x)
        x = x.reshape(batch_size, sequence_length, -1)
        if self.temporal_encoder is None:
            raise RuntimeError("temporal_encoder is required when sequence_length > 1")
        x = self.temporal_encoder(x)
        return self.decoder(x)
