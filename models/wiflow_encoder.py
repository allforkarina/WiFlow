from __future__ import annotations

import torch
from torch import nn

from .wiflow_encoder_asymmetric_cnn_layer2 import WiFlowEncoderAsymmetricCNNLayer2
from .wiflow_encoder_axial_attention_layer3 import WiFlowEncoderAxialAttentionLayer3
from .wiflow_encoder_tcn_layer1 import WiFlowEncoderTCNLayer1


class WiFlowEncoder(nn.Module):
    """WiFlow encoder that maps MM-Fi CSI features to [B, 64, 17, 10]."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = WiFlowEncoderTCNLayer1()
        self.layer2 = WiFlowEncoderAsymmetricCNNLayer2()
        self.layer3 = WiFlowEncoderAxialAttentionLayer3()

    def _prepare_axial_attention_input(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self._prepare_axial_attention_input(x)
        return self.layer3(x)
