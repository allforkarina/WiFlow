from __future__ import annotations

import torch
from torch import nn

from .skeleton import NUM_OPENPOSE_KEYPOINTS, build_normalized_adjacency


class WiFlowSkeletonDecoder(nn.Module):
    """Skeleton-aware decoder from [B, 256] pose embeddings to [B, 18, 2]."""

    def __init__(self) -> None:
        super().__init__()
        self.num_queries = NUM_OPENPOSE_KEYPOINTS
        self.embedding_dim = 256
        self.num_heads = 4
        self.joint_queries = nn.Parameter(torch.zeros(self.num_queries, self.embedding_dim))
        self.input_norm = nn.LayerNorm(self.embedding_dim)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        joint_query = self.joint_queries.unsqueeze(0).expand(batch_size, -1, -1)
        h = self.input_norm(joint_query + x.unsqueeze(1))

        gnn_output = torch.matmul(self.adjacency, self.gnn_projection(h))
        gnn_output = self.gnn_activation(gnn_output)
        h = self.gnn_norm(h + gnn_output)

        attention_output, _ = self.joint_attention(h, h, h, need_weights=False)
        h = self.attention_norm(h + attention_output)
        return self.coordinate_head(h)
