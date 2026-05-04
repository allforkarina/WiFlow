from __future__ import annotations

import torch
from torch import nn

from models import WiFlowJointDecoder


def test_wiflow_joint_decoder_output_shape() -> None:
    decoder = WiFlowJointDecoder()
    x = torch.randn(4, 256, 29, 10)

    y = decoder(x)

    assert y.shape == (4, 17, 2)


def test_wiflow_joint_decoder_configuration() -> None:
    decoder = WiFlowJointDecoder()

    assert decoder.num_queries == 17
    assert decoder.embedding_dim == 256
    assert decoder.joint_queries.shape == (17, 256)
    assert torch.equal(decoder.joint_queries, torch.zeros_like(decoder.joint_queries))
    assert decoder.adjacency.shape == (17, 17)
    assert "adjacency" in dict(decoder.named_buffers())
    assert "adjacency" not in dict(decoder.named_parameters())
    assert isinstance(decoder.cross_attention, nn.MultiheadAttention)
    assert decoder.cross_attention.embed_dim == 256
    assert decoder.cross_attention.num_heads == 8
    assert isinstance(decoder.joint_attention, nn.MultiheadAttention)
    assert decoder.joint_attention.embed_dim == 256
    assert decoder.joint_attention.num_heads == 4
    assert isinstance(decoder.coordinate_head[0], nn.Linear)
    assert decoder.coordinate_head[0].in_features == 256
    assert decoder.coordinate_head[0].out_features == 128
    assert decoder.coordinate_head[2].out_features == 2


def test_wiflow_joint_decoder_flattens_spatial_tokens() -> None:
    decoder = WiFlowJointDecoder()
    x = torch.randn(2, 256, 29, 10)

    tokens = decoder.flatten_tokens(x)

    assert tokens.shape == (2, 290, 256)
