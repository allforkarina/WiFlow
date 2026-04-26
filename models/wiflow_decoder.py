from __future__ import annotations

import torch
from torch import nn


class TemporalAttentionPooling(nn.Module):
    """Learn temporal weights for each keypoint feature."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention_logits = nn.Conv2d(
            in_channels=channels,
            out_channels=1,
            kernel_size=1,
        )

    def compute_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.attention_logits(x)
        return torch.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.compute_attention_weights(x)
        return torch.sum(x * weights, dim=-1, keepdim=True)


class WiFlowDecoder(nn.Module):
    """Joint-aware decoder that maps [B, 64, 17, 10] to [B, 17, 2]."""

    def __init__(self, num_x_bins: int = 128, num_y_bins: int = 128) -> None:
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
        self.temporal_pool = TemporalAttentionPooling(channels=32)
        self.joint_embedding = nn.Parameter(torch.zeros(17, 32))
        self.x_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, num_x_bins),
        )
        self.y_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, num_y_bins),
        )
        self.register_buffer("x_bin_centers", torch.linspace(0.0, 1.0, num_x_bins))
        self.register_buffer("y_bin_centers", torch.linspace(0.0, 1.0, num_y_bins))

    def decode_coordinate_distribution(
        self,
        x_logits: torch.Tensor,
        y_logits: torch.Tensor,
    ) -> torch.Tensor:
        x_probs = torch.softmax(x_logits, dim=-1)
        y_probs = torch.softmax(y_logits, dim=-1)
        x_coordinate = torch.sum(x_probs * self.x_bin_centers.view(1, 1, -1), dim=-1)
        y_coordinate = torch.sum(y_probs * self.y_bin_centers.view(1, 1, -1), dim=-1)
        return torch.stack((x_coordinate, y_coordinate), dim=-1)

    def forward_with_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.refinement(x)
        x = self.temporal_pool(x)
        x = x.squeeze(-1).transpose(1, 2)
        x = x + self.joint_embedding.unsqueeze(0)
        x_logits = self.x_head(x)
        y_logits = self.y_head(x)
        return self.decode_coordinate_distribution(x_logits, y_logits), x_logits, y_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction, _, _ = self.forward_with_logits(x)
        return prediction
