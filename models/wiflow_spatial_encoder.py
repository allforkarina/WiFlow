from __future__ import annotations

import torch
from torch import nn


class AsymmetricResidualDownsampleBlock(nn.Module):
    """Convolution block that downsamples only the subcarrier axis."""

    def __init__(self, in_channels: int, out_channels: int, spatial_stride: int) -> None:
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 3),
                stride=(1, spatial_stride),
                padding=(0, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=(1, spatial_stride),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main_path(x) + self.shortcut(x))


class WiFlowSpatialEncoder(nn.Module):
    """Spatial CSI encoder from [B, C, 114, 10] to [B, 128, 29, 10]."""

    def __init__(self, input_channels: int = 6) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.stem_channels = 32
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=self.stem_channels,
                kernel_size=(3, 5),
                stride=(1, 1),
                padding=(1, 2),
                bias=False,
            ),
            nn.BatchNorm2d(self.stem_channels),
            nn.ReLU(inplace=True),
        )
        self.resblock1 = AsymmetricResidualDownsampleBlock(32, 64, spatial_stride=2)
        self.resblock2 = AsymmetricResidualDownsampleBlock(64, 128, spatial_stride=2)
        self.resblock3 = AsymmetricResidualDownsampleBlock(128, 128, spatial_stride=1)

    def _to_conv_layout(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 3, 2)

    def _to_model_layout(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_conv_layout(x)               # [B, C, 10, 114]
        x = self.stem(x)                          # [B, 32, 10, 114]
        x = self.resblock1(x)                     # [B, 64, 10, 57]
        x = self.resblock2(x)                     # [B, 128, 10, 29]
        x = self.resblock3(x)                     # [B, 128, 10, 29]
        return self._to_model_layout(x)           # [B, 128, 29, 10]
