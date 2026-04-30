from __future__ import annotations

import torch
from torch import nn


class WiFlowEncoderAxialAttentionLayer3(nn.Module):
    """Third encoder layer that preserves [B, 64, 29, 10] axial features."""

    def __init__(self) -> None:
        super().__init__()
        self.channels = 64  # [B, 64, 10, 29] -> batch_size, channel, temporal, spatial
        self.num_heads = 8  # [B, 8, 10, 29]

        # [B, 64, 29, 10] -> [B x 29, 10, 64], extract the temporal features of one spatial token
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.channels,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.temporal_norm = nn.LayerNorm(self.channels)

        # [B, 64, 29, 10] -> [B x 10, 29, 64], exchange information across spatial tokens
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=self.channels,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.spatial_norm = nn.LayerNorm(self.channels)

    def _prepare_temporal_attention_input(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, spatial_tokens, temporal = x.shape                             # [B, 64, 29, 10]
        return x.permute(0, 2, 3, 1).reshape(batch_size * spatial_tokens, temporal, channels)

    def _restore_temporal_attention_output(
        self,
        x: torch.Tensor,
        batch_size: int,
        spatial_tokens: int,
        temporal: int,
    ) -> torch.Tensor:
        return x.reshape(batch_size, spatial_tokens, temporal, self.channels).permute(0, 3, 1, 2)

    def _prepare_spatial_attention_input(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, spatial_tokens, temporal = x.shape                             # [B, 64, 29, 10]
        return x.permute(0, 3, 2, 1).reshape(batch_size * temporal, spatial_tokens, channels)

    def _restore_spatial_attention_output(
        self,
        x: torch.Tensor,
        batch_size: int,
        spatial_tokens: int,
        temporal: int,
    ) -> torch.Tensor:
        return x.reshape(batch_size, temporal, spatial_tokens, self.channels).permute(0, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, spatial_tokens, temporal = x.shape

        temporal_input = self._prepare_temporal_attention_input(x)                  # [B x 29, 10, 64]
        temporal_output, _ = self.temporal_attention(
            temporal_input,
            temporal_input,
            temporal_input,
            need_weights=False,
        )
        temporal_output = self.temporal_norm(temporal_output + temporal_input)      # Residual connection
        x = self._restore_temporal_attention_output(                                # [B x 29, 10, 64] -> [B, 64, 29, 10]
            temporal_output,
            batch_size,
            spatial_tokens,
            temporal,
        )

        #^ problem 02: do temporal attention first, then do spatial attention on the processed feature.
        spatial_input = self._prepare_spatial_attention_input(x)                    # [B x 10, 29, 64]
        spatial_output, _ = self.spatial_attention(
            spatial_input,
            spatial_input,
            spatial_input,
            need_weights=False,
        )
        spatial_output = self.spatial_norm(spatial_output + spatial_input)          # Residual connection
        return self._restore_spatial_attention_output(                              # [B x 10, 29, 64] -> [B, 64, 29, 10]
            spatial_output,
            batch_size,
            spatial_tokens,
            temporal,
        )
