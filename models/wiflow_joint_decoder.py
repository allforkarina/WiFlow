from __future__ import annotations

import torch
from torch import nn

from .skeleton import NUM_COCO_KEYPOINTS, build_normalized_adjacency


class WiFlowJointDecoder(nn.Module):
    """Decode COCO17 coordinates from [B, 256, 29, 10] feature maps."""

    def __init__(self) -> None:
        super().__init__()
        self.num_queries = NUM_COCO_KEYPOINTS
        self.embedding_dim = 256
        self.cross_attention_heads = 8
        self.num_heads = 4
        self.joint_queries = nn.Parameter(torch.zeros(self.num_queries, self.embedding_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.cross_attention_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.cross_attention_norm = nn.LayerNorm(self.embedding_dim)
        self.gnn_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.gnn_activation = nn.SiLU()
        self.gnn_norm = nn.LayerNorm(self.embedding_dim)
        self.joint_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.attention_norm = nn.LayerNorm(self.embedding_dim)
        self.coordinate_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 2),
        )
        self.register_buffer("adjacency", build_normalized_adjacency(), persistent=False)

    def flatten_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("WiFlowJointDecoder expects input shaped [B, 256, 29, 10]")
        return x.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.flatten_tokens(x)
        joint_queries = self.joint_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)

        attended, _ = self.cross_attention(joint_queries, tokens, tokens, need_weights=False)
        h = self.cross_attention_norm(attended + joint_queries)

        gnn_output = torch.matmul(self.adjacency, self.gnn_projection(h))
        gnn_output = self.gnn_activation(gnn_output)
        h = self.gnn_norm(h + gnn_output)

        attention_output, _ = self.joint_attention(h, h, h, need_weights=False)
        h = self.attention_norm(h + attention_output)
        return self.coordinate_head(h)
