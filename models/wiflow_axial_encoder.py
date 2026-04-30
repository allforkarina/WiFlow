from __future__ import annotations

import torch
from torch import nn


class WiFlowAxialEncoder(nn.Module):
    """Axial attention encoder from [B, 128, 29, 10] to [B, 256, 29, 10]."""

    def __init__(self) -> None:
        super().__init__()
        self.input_channels = 128
        self.output_channels = 256
        self.num_heads = 8
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=self.input_channels,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.spatial_norm = nn.LayerNorm(self.input_channels)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.input_channels,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.temporal_norm = nn.LayerNorm(self.input_channels)
        self.channel_projection = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=1)

    def _prepare_spatial_attention_input(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, spatial_tokens, temporal = x.shape
        return x.permute(0, 3, 2, 1).reshape(batch_size * temporal, spatial_tokens, channels)

    def _restore_spatial_attention_output(
        self,
        x: torch.Tensor,
        batch_size: int,
        spatial_tokens: int,
        temporal: int,
    ) -> torch.Tensor:
        return x.reshape(batch_size, temporal, spatial_tokens, self.input_channels).permute(0, 3, 2, 1)

    def _prepare_temporal_attention_input(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, spatial_tokens, temporal = x.shape
        return x.permute(0, 2, 3, 1).reshape(batch_size * spatial_tokens, temporal, channels)

    def _restore_temporal_attention_output(
        self,
        x: torch.Tensor,
        batch_size: int,
        spatial_tokens: int,
        temporal: int,
    ) -> torch.Tensor:
        return x.reshape(batch_size, spatial_tokens, temporal, self.input_channels).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, spatial_tokens, temporal = x.shape

        spatial_input = self._prepare_spatial_attention_input(x)
        spatial_output, _ = self.spatial_attention(
            spatial_input,
            spatial_input,
            spatial_input,
            need_weights=False,
        )
        spatial_output = self.spatial_norm(spatial_output + spatial_input)
        x = self._restore_spatial_attention_output(spatial_output, batch_size, spatial_tokens, temporal)

        temporal_input = self._prepare_temporal_attention_input(x)
        temporal_output, _ = self.temporal_attention(
            temporal_input,
            temporal_input,
            temporal_input,
            need_weights=False,
        )
        temporal_output = self.temporal_norm(temporal_output + temporal_input)
        x = self._restore_temporal_attention_output(temporal_output, batch_size, spatial_tokens, temporal)
        return self.channel_projection(x)
