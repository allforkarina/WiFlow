from __future__ import annotations

import torch
from torch import nn


class TemporalAttentionPooling(nn.Module):
    """Learn temporal weights for each keypoint feature."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention_logits = nn.Conv2d(
            in_channels=channels,
            out_channels=1,
            kernel_size=1,
        )

    def compute_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.attention_logits(x)
        return torch.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.compute_attention_weights(x)
        return torch.sum(x * weights, dim=-1, keepdim=True)


class WiFlowDecoder(nn.Module):
    """Joint-aware decoder that maps [B, 64, 17, 10] to [B, 17, 2]."""

    def __init__(self) -> None:
        super().__init__()
        self.refinement = nn.Sequential(                        # 64 channels -> 32 channels
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.temporal_pool = TemporalAttentionPooling(channels=32)
        self.joint_embedding = nn.Parameter(torch.zeros(17, 32))
        self.coordinate_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.refinement(x)
        x = self.temporal_pool(x)
        x = x.squeeze(-1).transpose(1, 2)
        x = x + self.joint_embedding.unsqueeze(0)
        return self.coordinate_head(x)
