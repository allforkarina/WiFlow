from __future__ import annotations

import torch
from torch import nn

from .wiflow_decoder import WiFlowDecoder
from .wiflow_encoder import WiFlowEncoder


class WiFlowModel(nn.Module):
    """End-to-end WiFlow model that maps [B, 3, 114, 10] to [B, 17, 2]."""

    def __init__(self, num_x_bins: int = 128, num_y_bins: int = 128) -> None:
        super().__init__()
        self.encoder = WiFlowEncoder()
        self.decoder = WiFlowDecoder(num_x_bins=num_x_bins, num_y_bins=num_y_bins)

    def forward_with_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        return self.decoder.forward_with_logits(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction, _, _ = self.forward_with_logits(x)
        return prediction
