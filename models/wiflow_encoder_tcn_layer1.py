from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class WiFlowEncoderTCNLayer1(nn.Module):
    """First encoder TCN layer for MM-Fi inputs shaped [B, 342, 10]."""

    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 342                          # flatten the antenna and subcarrier dimensions, 3 x 114
        self.temporal_channels = 342                    # tcn only operates on the temporal dimension
        self.out_channels = 340                         # reduce the number of channels to 340 to match the asymmetric convolution
        self.kernel_size = 3                            # kernel size
        self.dilation = 1                               # dilation factor, s - k*d
        self.temporal_conv = nn.Conv1d(
            in_channels=self.in_channels,               # temporal convolution dont change the feature dim
            out_channels=self.temporal_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
        )
        self.channel_projection = nn.Conv1d(            # last layer of match the size
            in_channels=self.temporal_channels,
            out_channels=self.out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left_padding = (self.kernel_size - 1) * self.dilation   # left padding for causal convolution, s = 0, 1 need the padding
        x = F.pad(x, (left_padding, 0))                         # pad on the temporal size
        x = self.temporal_conv(x)                               # temporal convolution
        return self.channel_projection(x)                       # project to the desired number of channels
