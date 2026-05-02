from __future__ import annotations

import torch
from torch import nn


class AsymmetricResidualDownsampleBlock(nn.Module):
    """Time-frequency convolution block that downsamples only the subcarrier axis."""

    def __init__(self, in_channels: int, out_channels: int, spatial_stride: int) -> None:
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, spatial_stride),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
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
        if input_channels <= 0 or input_channels % 3 != 0:
            raise ValueError("input_channels must be a positive multiple of 3 for three-antenna CSI features")
        self.input_channels = input_channels
        self.num_features = input_channels // 3
        self.stem_channels = 32

        stem_channels_by_feature = self._split_stem_channels(self.stem_channels, self.num_features)
        self.antenna_mixers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3, 3, kernel_size=1, bias=False),
                    nn.BatchNorm2d(3),
                    nn.ReLU(inplace=True),
                )
                for _ in range(self.num_features)
            ]
        )
        self.feature_stems = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=feature_channels,
                        kernel_size=(3, 5),
                        stride=(1, 1),
                        padding=(1, 2),
                        bias=False,
                    ),
                    nn.BatchNorm2d(feature_channels),
                    nn.ReLU(inplace=True),
                )
                for feature_channels in stem_channels_by_feature
            ]
        )
        self.resblock1 = AsymmetricResidualDownsampleBlock(32, 64, spatial_stride=2)
        self.resblock2 = AsymmetricResidualDownsampleBlock(64, 128, spatial_stride=2)
        self.resblock3 = AsymmetricResidualDownsampleBlock(128, 128, spatial_stride=1)

    def _split_stem_channels(self, total_channels: int, num_features: int) -> list[int]:
        base_channels = total_channels // num_features
        remainder = total_channels % num_features
        return [
            base_channels + (1 if feature_index < remainder else 0)
            for feature_index in range(num_features)
        ]

    def _to_conv_layout(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 3, 2)

    def _to_model_layout(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(2, 3)

    def _apply_feature_stems(self, x: torch.Tensor) -> torch.Tensor:
        feature_outputs = []
        for feature_index, (antenna_mixer, feature_stem) in enumerate(
            zip(self.antenna_mixers, self.feature_stems)
        ):
            start_channel = feature_index * 3
            feature_input = x[:, start_channel : start_channel + 3]
            mixed_feature = antenna_mixer(feature_input)
            feature_outputs.append(feature_stem(mixed_feature))
        return torch.cat(feature_outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_conv_layout(x)               # [B, C, 10, 114]
        x = self._apply_feature_stems(x)          # [B, 32, 10, 114]
        x = self.resblock1(x)                     # [B, 64, 10, 57]
        x = self.resblock2(x)                     # [B, 128, 10, 29]
        x = self.resblock3(x)                     # [B, 128, 10, 29]
        return self._to_model_layout(x)           # [B, 128, 29, 10]
