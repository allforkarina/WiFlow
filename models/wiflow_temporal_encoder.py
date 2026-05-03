from __future__ import annotations

import torch
from torch import nn


class WiFlowTemporalEncoder(nn.Module):
    """Temporal self-attention from [B, N, 256] to the middle [B, 256] token."""

    def __init__(self, sequence_length: int, embedding_dim: int = 256) -> None:
        super().__init__()
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = 8
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim))
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    @property
    def middle_index(self) -> int:
        return self.sequence_length // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("WiFlowTemporalEncoder expects input shaped [B, N, 256]")
        if x.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, got {x.shape[1]}"
            )
        if x.shape[2] != self.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {x.shape[2]}")

        tokens = x + self.position_embedding
        attended, _ = self.self_attention(tokens, tokens, tokens, need_weights=False)
        encoded = self.output_norm(attended + tokens)
        return encoded[:, self.middle_index]
