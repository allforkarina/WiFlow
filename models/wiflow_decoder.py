from __future__ import annotations

import torch
from torch import nn


class WiFlowDecoder(nn.Module):
    """Joint-query decoder that maps [B, 290, 64] CSI tokens to [B, 17, 2]."""

    def __init__(self) -> None:
        super().__init__()
        self.num_queries = 17           # 17 keypoints, each refer to one query token
        self.embedding_dim = 64         # embedding dimension, same as the channel dimension of the input tokens.
        self.num_heads = 8              # attention heads.
        self.joint_queries = nn.Parameter(torch.zeros(self.num_queries, self.embedding_dim))    # [17, 64]
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.attention_norm = nn.LayerNorm(self.embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.ffn_norm = nn.LayerNorm(self.embedding_dim)
        self.coordinate_head = nn.Linear(self.embedding_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        query = self.joint_queries.unsqueeze(0).expand(batch_size, -1, -1)                  # [B, 17, 64]
        attention_output, _ = self.cross_attention(query, x, x, need_weights=False)         # 17 query to learn the input feature of csi with self-attention
        query = self.attention_norm(query + attention_output)                               # [B, 17, 64]   
        ffn_output = self.ffn(query)                                                        # two ffn, channel unchanged with more weights
        query = self.ffn_norm(query + ffn_output)                                           # [B, 17, 64] 
        return self.coordinate_head(query)                                                  # [B, 17, 2], channel to coordinate X and Y.
