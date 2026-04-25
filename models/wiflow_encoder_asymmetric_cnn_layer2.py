from __future__ import annotations

import torch
from torch import nn


class AsymmetricResidualDownsampleBlock(nn.Module):
    """Convolution block with asymmetric kernels, used to contract the spatial dimension"""

    def __init__(self, in_channels: int, out_channels: int, spatial_stride: int) -> None:
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(                              
                in_channels=in_channels,            # [B, in_channels, 10, f]
                out_channels=out_channels,          # [B, out_channels, 10, f / stride]
                kernel_size=(1, 3),
                stride=(1, spatial_stride),         # temporal dim stride 1, spatial dim stride 2 or 5
                padding=(0, 1),                     # temporal dim no pad, spatial dim pad 1
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),           # BatchNorm
            nn.ReLU(inplace=True),                  # ReLU activation
            nn.Conv2d(                              
                in_channels=out_channels,           # [B, in_channels, 10, f / stride]
                out_channels=out_channels,          # [B, out_channels, 10, f / stride]
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),           # BatchNorm
        )

        # dowmsample to match the keypoints' num
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,            # [B, in_channels, 10, f]
                out_channels=out_channels,          # [B, out_channels, 10, f / stride]
                kernel_size=1,
                stride=(1, spatial_stride),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.ReLU(inplace=True)     # ReLU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection + Activation
        return self.activation(self.main_path(x) + self.shortcut(x))


class WiFlowEncoderAsymmetricCNNLayer2(nn.Module):
    """Structured CSI encoder that maps [B, 3, 114, 10] to [B, 64, 10, 17]."""

    def __init__(self) -> None:
        super().__init__()
        self.input_channels = 3                         # antenna channels
        self.input_temporal = 10
        self.input_spatial = 114
        self.stem_channels = 16                         # [B, 16, 10, 114]
        self.stem = nn.Sequential(                      # 3 -> 16 channels
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=self.stem_channels,
                kernel_size=(3, 5),
                stride=(1, 1),
                padding=(1, 2),                         # preserve temporal and subcarrier dimensions
                bias=False,
            ),
            nn.BatchNorm2d(self.stem_channels),
            nn.ReLU(inplace=True),
        )
        self.resblock1 = AsymmetricResidualDownsampleBlock(
            in_channels=16,
            out_channels=32,
            spatial_stride=2,                           # stride = 2, 114 -> 57
        )
        self.resblock2 = AsymmetricResidualDownsampleBlock(
            in_channels=32,
            out_channels=64,
            spatial_stride=2,                           # stride = 2, 57 -> 29
        )
        self.resblock3 = AsymmetricResidualDownsampleBlock(
            in_channels=64,
            out_channels=64,
            spatial_stride=1,                           # refine structured CSI features
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((10, 17))

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 3, 2)                   # [B, 3, 114, 10] -> [B, 3, 10, 114]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._reshape_input(x)
        x = self.stem(x)                                # [B, 16, 10, 114]
        x = self.resblock1(x)                           # [B, 32, 10, 57]
        x = self.resblock2(x)                           # [B, 64, 10, 29]
        x = self.resblock3(x)                           # [B, 64, 10, 29]
        return self.spatial_pool(x)                     # [B, 64, 10, 17]
