from __future__ import annotations

import torch
from torch import nn

from .wiflow_decoder import WiFlowDecoder
from .wiflow_encoder import WiFlowEncoder


class WiFlowModel(nn.Module):
    """End-to-end WiFlow model that maps [B, 3, 114, 10] to [B, 17, 2]."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = WiFlowEncoder()
        self.decoder = WiFlowDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.encoder.flatten_tokens(x)
        return self.decoder(x)
