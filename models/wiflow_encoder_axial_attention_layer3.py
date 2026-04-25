from __future__ import annotations

import torch
from torch import nn


class WiFlowEncoderAxialAttentionLayer3(nn.Module):
    """Third encoder layer that preserves [B, 64, 17, 10] axial features."""

    def __init__(self) -> None:
        super().__init__()
        self.channels = 64  # [B, 64, 10, 17] -> batch_size, channel, temporal, spatial
        self.num_heads = 8  # [B, 8, 10, 17]

        # [B, 64, 17, 10] -> [B x 10, 64, 17], extract the keypoints' features in the same time shot
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.channels,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.temporal_norm = nn.LayerNorm(self.channels)

        # [B, 64, 17, 10] -> [B x 17, 64, 10], extract the temporal features of the same keypoint
        self.keypoint_attention = nn.MultiheadAttention(
            embed_dim=self.channels,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.keypoint_norm = nn.LayerNorm(self.channels)

    def _prepare_temporal_attention_input(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, keypoints, temporal = x.shape                                 # [B, 64, 17, 10]
        return x.permute(0, 2, 3, 1).reshape(batch_size * keypoints, temporal, channels)    # [B x 17, 10, 64]

    def _restore_temporal_attention_output(
        self,
        x: torch.Tensor,
        batch_size: int,
        keypoints: int,
        temporal: int,
    ) -> torch.Tensor:
        return x.reshape(batch_size, keypoints, temporal, self.channels).permute(0, 3, 1, 2)

    def _prepare_keypoint_attention_input(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, keypoints, temporal = x.shape                                 # [B, 64, 17, 10]
        return x.permute(0, 3, 2, 1).reshape(batch_size * temporal, keypoints, channels)    # [B x 10, 17, 64]

    def _restore_keypoint_attention_output(
        self,
        x: torch.Tensor,
        batch_size: int,
        keypoints: int,
        temporal: int,
    ) -> torch.Tensor:
        return x.reshape(batch_size, temporal, keypoints, self.channels).permute(0, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, keypoints, temporal = x.shape

        temporal_input = self._prepare_temporal_attention_input(x)                  # [B x 17, 10, 64]
        temporal_output, _ = self.temporal_attention(
            temporal_input,
            temporal_input,
            temporal_input,
            need_weights=False,
        )
        temporal_output = self.temporal_norm(temporal_output + temporal_input)      # Residual connection
        x = self._restore_temporal_attention_output(                                # [B x 17, 10, 64] -> [B, 64, 17, 10]
            temporal_output,
            batch_size,
            keypoints,
            temporal,
        )

        keypoint_input = self._prepare_keypoint_attention_input(x)                  # [B x 10, 17, 64]
        keypoint_output, _ = self.keypoint_attention(
            keypoint_input,
            keypoint_input,
            keypoint_input,
            need_weights=False,
        )
        keypoint_output = self.keypoint_norm(keypoint_output + keypoint_input)      # Residual connection
        return self._restore_keypoint_attention_output(                             # [B x 10, 17, 64] -> [B, 64, 17, 10]
            keypoint_output,
            batch_size,
            keypoints,
            temporal,
        )
