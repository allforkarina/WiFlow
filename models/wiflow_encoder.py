from __future__ import annotations

import torch
from torch import nn

from .wiflow_encoder_asymmetric_cnn_layer2 import WiFlowEncoderAsymmetricCNNLayer2
from .wiflow_encoder_axial_attention_layer3 import WiFlowEncoderAxialAttentionLayer3


class WiFlowEncoder(nn.Module):
    """WiFlow encoder that maps MM-Fi CSI features to [B, 64, 29, 10]."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = WiFlowEncoderAsymmetricCNNLayer2()
        self.layer3 = WiFlowEncoderAxialAttentionLayer3()

    def _prepare_axial_attention_input(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(2, 3)

    def flatten_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, spatial_tokens, temporal = x.shape
        return x.permute(0, 2, 3, 1).reshape(batch_size, spatial_tokens * temporal, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)                                  # [B, 64, 10, 29]
        x = self._prepare_axial_attention_input(x)          # [B, 64, 29, 10]
        return self.layer3(x)                               # [B, 64, 29, 10]
