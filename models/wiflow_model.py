from __future__ import annotations

import torch
from torch import nn

from .wiflow_axial_encoder import WiFlowAxialEncoder
from .wiflow_heatmap_decoder import WiFlowMSFNDecoder
from .wiflow_hierarchical_joint_decoder import WiFlowHierarchicalJointDecoder
from .wiflow_joint_decoder import WiFlowJointDecoder
from .wiflow_spatial_encoder import WiFlowSpatialEncoder
from .wiflow_spatial_temporal_fuser import WiFlowSpatialTemporalFuser
from pose_targets import decode_pcm_argmax

DECODER_TYPES = ("joint", "hierarchical", "heatmap_msfn")


class WiFlowModel(nn.Module):
    """End-to-end WiFlow model that maps CSI features to COCO17 coordinates."""

    def __init__(
        self,
        input_channels: int = 6,
        axial_mode: str = "spatial_then_temporal",
        sequence_length: int = 1,
        decoder_type: str = "joint",
        heatmap_size: int = 36,
    ) -> None:
        super().__init__()
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")
        if decoder_type not in DECODER_TYPES:
            raise ValueError(f"decoder_type must be one of {DECODER_TYPES}")
        self.input_channels = input_channels
        self.axial_mode = axial_mode
        self.sequence_length = sequence_length
        self.decoder_type = decoder_type
        self.heatmap_size = heatmap_size
        self.spatial_encoder = WiFlowSpatialEncoder(input_channels=input_channels)
        self.axial_encoder = WiFlowAxialEncoder(mode=axial_mode)
        self.temporal_fuser = (
            WiFlowSpatialTemporalFuser(sequence_length=sequence_length)
            if sequence_length > 1
            else None
        )
        if decoder_type == "joint":
            self.decoder = WiFlowJointDecoder()
        elif decoder_type == "hierarchical":
            self.decoder = WiFlowHierarchicalJointDecoder()
        else:
            self.decoder = WiFlowMSFNDecoder(heatmap_size=heatmap_size)

    def decode_features(self, x: torch.Tensor):
        decoder_output = self.decoder(x)
        if self.decoder_type != "heatmap_msfn":
            return decoder_output
        stages = decoder_output
        keypoints = decode_pcm_argmax(stages[-1]["pcm"])
        return {"keypoints": keypoints, "stages": stages}

    def forward(self, x: torch.Tensor):
        if self.sequence_length == 1:
            if x.ndim != 4:
                raise ValueError("single-frame WiFlowModel expects input shaped [B, C, 114, 10]")
            x = self.spatial_encoder(x)
            x = self.axial_encoder(x)
            return self.decode_features(x)

        if x.ndim != 5:
            raise ValueError("temporal WiFlowModel expects input shaped [B, N, C, 114, 10]")
        if x.shape[1] != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {x.shape[1]}")

        batch_size, sequence_length, channels, subcarriers, frames = x.shape
        x = x.reshape(batch_size * sequence_length, channels, subcarriers, frames)
        x = self.spatial_encoder(x)
        x = self.axial_encoder(x)
        x = x.reshape(batch_size, sequence_length, x.shape[1], x.shape[2], x.shape[3])
        if self.temporal_fuser is None:
            raise RuntimeError("temporal_fuser is required when sequence_length > 1")
        x = self.temporal_fuser(x)
        return self.decode_features(x)
