from __future__ import annotations

import torch
from torch import nn

from models import WiFlowEncoderAxialAttentionLayer3


def test_wiflow_encoder_axial_attention_layer3_preserves_shape() -> None:
    layer = WiFlowEncoderAxialAttentionLayer3()
    x = torch.randn(4, 64, 17, 10)

    y = layer(x)

    assert y.shape == (4, 64, 17, 10)


def test_wiflow_encoder_axial_attention_layer3_supports_single_item_batch() -> None:
    layer = WiFlowEncoderAxialAttentionLayer3()
    x = torch.randn(1, 64, 17, 10)

    y = layer(x)

    assert y.shape == (1, 64, 17, 10)


def test_wiflow_encoder_axial_attention_layer3_attention_configuration() -> None:
    layer = WiFlowEncoderAxialAttentionLayer3()

    assert layer.temporal_attention.embed_dim == 64
    assert layer.temporal_attention.num_heads == 8
    assert layer.keypoint_attention.embed_dim == 64
    assert layer.keypoint_attention.num_heads == 8
    assert isinstance(layer.temporal_norm, nn.LayerNorm)
    assert isinstance(layer.keypoint_norm, nn.LayerNorm)
    assert layer.temporal_norm.normalized_shape == (64,)
    assert layer.keypoint_norm.normalized_shape == (64,)


def test_wiflow_encoder_axial_attention_layer3_reshape_contracts() -> None:
    layer = WiFlowEncoderAxialAttentionLayer3()
    x = torch.randn(2, 64, 17, 10)

    temporal_input = layer._prepare_temporal_attention_input(x)
    temporal_restored = layer._restore_temporal_attention_output(
        temporal_input,
        batch_size=2,
        keypoints=17,
        temporal=10,
    )
    keypoint_input = layer._prepare_keypoint_attention_input(x)
    keypoint_restored = layer._restore_keypoint_attention_output(
        keypoint_input,
        batch_size=2,
        keypoints=17,
        temporal=10,
    )

    assert temporal_input.shape == (34, 10, 64)
    assert temporal_restored.shape == (2, 64, 17, 10)
    assert keypoint_input.shape == (20, 17, 64)
    assert keypoint_restored.shape == (2, 64, 17, 10)
    assert torch.equal(temporal_restored, x)
    assert torch.equal(keypoint_restored, x)
