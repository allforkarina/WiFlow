from __future__ import annotations

import torch
from torch import nn


class WiFlowAttentionPooler(nn.Module):
    """Attention pool CSI tokens from [B, 256, 29, 10] to [B, 256]."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding_dim = 256
        self.num_heads = 8
        self.global_query = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.output_norm = nn.LayerNorm(self.embedding_dim)

    def flatten_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, spatial_tokens, temporal = x.shape
        return x.permute(0, 2, 3, 1).reshape(batch_size, spatial_tokens * temporal, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.flatten_tokens(x)
        query = self.global_query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.cross_attention(query, tokens, tokens, need_weights=False)
        return self.output_norm(pooled + query).squeeze(1)
