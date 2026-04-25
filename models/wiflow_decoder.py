from __future__ import annotations

import torch
from torch import nn


class WiFlowDecoder(nn.Module):
    """Lightweight decoder that maps [B, 64, 17, 10] to [B, 17, 2]."""

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
        self.coordinate_projection = nn.Conv2d(                 # 32 channels -> 2 channels (x, y coordinates)
            in_channels=32,
            out_channels=2,
            kernel_size=1,
        )
        self.temporal_pool = nn.AdaptiveAvgPool2d((17, 1))      # average pool the temporal dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.refinement(x)
        x = self.coordinate_projection(x)
        x = self.temporal_pool(x)
        x = x.squeeze(-1)
        return x.transpose(1, 2)
