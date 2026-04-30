from __future__ import annotations

import torch
from torch import nn

from models import WiFlowAxialEncoder


def test_wiflow_axial_encoder_output_shape() -> None:
    layer = WiFlowAxialEncoder()
    x = torch.randn(4, 128, 29, 10)

    y = layer(x)

    assert y.shape == (4, 256, 29, 10)


def test_wiflow_axial_encoder_attention_configuration() -> None:
    layer = WiFlowAxialEncoder()

    assert layer.spatial_attention.embed_dim == 128
    assert layer.spatial_attention.num_heads == 8
    assert layer.temporal_attention.embed_dim == 128
    assert layer.temporal_attention.num_heads == 8
    assert isinstance(layer.spatial_norm, nn.LayerNorm)
    assert isinstance(layer.temporal_norm, nn.LayerNorm)
    assert layer.channel_projection.in_channels == 128
    assert layer.channel_projection.out_channels == 256


def test_wiflow_axial_encoder_reshape_contracts() -> None:
    layer = WiFlowAxialEncoder()
    x = torch.randn(2, 128, 29, 10)

    spatial_input = layer._prepare_spatial_attention_input(x)
    spatial_restored = layer._restore_spatial_attention_output(spatial_input, 2, 29, 10)
    temporal_input = layer._prepare_temporal_attention_input(x)
    temporal_restored = layer._restore_temporal_attention_output(temporal_input, 2, 29, 10)

    assert spatial_input.shape == (20, 29, 128)
    assert spatial_restored.shape == (2, 128, 29, 10)
    assert temporal_input.shape == (58, 10, 128)
    assert temporal_restored.shape == (2, 128, 29, 10)
    assert torch.equal(spatial_restored, x)
    assert torch.equal(temporal_restored, x)
