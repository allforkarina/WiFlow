from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CausalTCNBlock(nn.Module):
    """One causal temporal convolution block that preserves sequence length."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.kernel_size = 3                                                # kernel size
        self.dilation = dilation                                            # dilation factor
        self.left_padding = (self.kernel_size - 1) * dilation               # calculate left padding to preserve sequence length
        self.temporal_conv = nn.Conv1d(
            in_channels=in_channels,                                        # [B, 342, 10]
            out_channels=out_channels,                                      # [B, 342, 10] for the first three blocks, [B, 340, 10] for the last block
            kernel_size=self.kernel_size,                                   # kernel size 
            dilation=dilation,                                              # dilation factor (1, 2, 4, 8)
        )
        self.norm = nn.BatchNorm1d(out_channels)                            # batch normalization
        self.activation = nn.SiLU()                                         # activation function
        self.use_residual = in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.pad(x, (self.left_padding, 0))                                # pad for casual convolution
        x = self.temporal_conv(x)                                           # temporal convolution
        x = self.norm(x)
        x = self.activation(x)
        if self.use_residual:
            x = x + residual
        return x


class WiFlowEncoderTCNLayer1(nn.Module):
    """Encoder TCN stack for MM-Fi inputs shaped [B, 342, 10]."""

    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 342                          # flatten the antenna and subcarrier dimensions, 3 x 114
        self.temporal_channels = 342                    # tcn only operates on the temporal dimension
        self.out_channels = 340                         # reduce the number of channels to 340 to match the asymmetric convolution
        self.kernel_size = 3                            # kernel size
        self.dilations = (1, 2, 4, 8)                   # four TCN layers cover the short MM-Fi window
        self.blocks = nn.ModuleList(
            [
                CausalTCNBlock(self.in_channels, self.temporal_channels, dilation=1),
                CausalTCNBlock(self.temporal_channels, self.temporal_channels, dilation=2),
                CausalTCNBlock(self.temporal_channels, self.temporal_channels, dilation=4),
                CausalTCNBlock(self.temporal_channels, self.out_channels, dilation=8),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
