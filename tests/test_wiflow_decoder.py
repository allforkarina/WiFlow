from __future__ import annotations

import torch
from torch import nn

from models import WiFlowDecoder


def test_wiflow_decoder_output_shape() -> None:
    decoder = WiFlowDecoder()
    x = torch.randn(4, 290, 64)

    y = decoder(x)

    assert y.shape == (4, 17, 2)


def test_wiflow_decoder_supports_single_item_batch() -> None:
    decoder = WiFlowDecoder()
    x = torch.randn(1, 290, 64)

    y = decoder(x)

    assert y.shape == (1, 17, 2)


def test_wiflow_decoder_configuration() -> None:
    decoder = WiFlowDecoder()

    assert decoder.num_queries == 17
    assert decoder.embedding_dim == 64
    assert decoder.num_heads == 8
    assert decoder.joint_queries.shape == (17, 64)
    assert isinstance(decoder.cross_attention, nn.MultiheadAttention)
    assert decoder.cross_attention.embed_dim == 64
    assert decoder.cross_attention.num_heads == 8
    assert isinstance(decoder.attention_norm, nn.LayerNorm)
    assert isinstance(decoder.ffn[0], nn.Linear)
    assert decoder.ffn[0].in_features == 64
    assert decoder.ffn[0].out_features == 64
    assert isinstance(decoder.ffn[1], nn.SiLU)
    assert isinstance(decoder.ffn[2], nn.Linear)
    assert decoder.ffn[2].in_features == 64
    assert decoder.ffn[2].out_features == 64
    assert isinstance(decoder.ffn_norm, nn.LayerNorm)
    assert isinstance(decoder.coordinate_head, nn.Linear)
    assert decoder.coordinate_head.in_features == 64
    assert decoder.coordinate_head.out_features == 2


def test_wiflow_decoder_uses_no_adaptive_average_pooling() -> None:
    decoder = WiFlowDecoder()

    assert not any(isinstance(module, nn.AdaptiveAvgPool2d) for module in decoder.modules())


def test_wiflow_decoder_uses_joint_queries_and_cross_attention() -> None:
    decoder = WiFlowDecoder()

    assert decoder.joint_queries.requires_grad
    assert isinstance(decoder.cross_attention, nn.MultiheadAttention)
