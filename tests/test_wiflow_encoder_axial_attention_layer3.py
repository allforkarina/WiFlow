from __future__ import annotations

import torch
from torch import nn

from models import WiFlowEncoderAxialAttentionLayer3


def test_wiflow_encoder_axial_attention_layer3_preserves_shape() -> None:
    layer = WiFlowEncoderAxialAttentionLayer3()
    x = torch.randn(4, 64, 29, 10)

    y = layer(x)

    assert y.shape == (4, 64, 29, 10)


def test_wiflow_encoder_axial_attention_layer3_supports_single_item_batch() -> None:
    layer = WiFlowEncoderAxialAttentionLayer3()
    x = torch.randn(1, 64, 29, 10)

    y = layer(x)

    assert y.shape == (1, 64, 29, 10)


def test_wiflow_encoder_axial_attention_layer3_attention_configuration() -> None:
    layer = WiFlowEncoderAxialAttentionLayer3()

    assert layer.temporal_attention.embed_dim == 64
    assert layer.temporal_attention.num_heads == 8
    assert layer.spatial_attention.embed_dim == 64
    assert layer.spatial_attention.num_heads == 8
    assert isinstance(layer.temporal_norm, nn.LayerNorm)
    assert isinstance(layer.spatial_norm, nn.LayerNorm)
    assert layer.temporal_norm.normalized_shape == (64,)
    assert layer.spatial_norm.normalized_shape == (64,)


def test_wiflow_encoder_axial_attention_layer3_reshape_contracts() -> None:
    layer = WiFlowEncoderAxialAttentionLayer3()
    x = torch.randn(2, 64, 29, 10)

    temporal_input = layer._prepare_temporal_attention_input(x)
    temporal_restored = layer._restore_temporal_attention_output(
        temporal_input,
        batch_size=2,
        spatial_tokens=29,
        temporal=10,
    )
    spatial_input = layer._prepare_spatial_attention_input(x)
    spatial_restored = layer._restore_spatial_attention_output(
        spatial_input,
        batch_size=2,
        spatial_tokens=29,
        temporal=10,
    )

    assert temporal_input.shape == (58, 10, 64)
    assert temporal_restored.shape == (2, 64, 29, 10)
    assert spatial_input.shape == (20, 29, 64)
    assert spatial_restored.shape == (2, 64, 29, 10)
    assert torch.equal(temporal_restored, x)
    assert torch.equal(spatial_restored, x)
