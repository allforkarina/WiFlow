from __future__ import annotations

import torch
from torch import nn

from models import WiFlowHierarchicalJointDecoder


def test_wiflow_hierarchical_joint_decoder_output_shape() -> None:
    decoder = WiFlowHierarchicalJointDecoder()
    x = torch.randn(2, 256, 29, 16)

    y = decoder(x)

    assert y.shape == (2, 18, 2)


def test_wiflow_hierarchical_joint_decoder_configuration() -> None:
    decoder = WiFlowHierarchicalJointDecoder()

    assert decoder.num_queries == 18
    assert decoder.embedding_dim == 256
    assert decoder.joint_queries.shape == (18, 256)
    assert torch.equal(decoder.joint_queries, torch.zeros_like(decoder.joint_queries))
    assert decoder.stage_indices == (
        (0, 1, 2, 5, 8, 11),
        (3, 4, 6, 7, 9, 10, 12, 13),
        (14, 15, 16, 17),
    )
    flattened_indices = [joint for stage in decoder.stage_indices for joint in stage]
    assert decoder.stage_order == tuple(flattened_indices)
    assert decoder.openpose_order == tuple(flattened_indices.index(joint) for joint in range(18))
    assert sorted(flattened_indices) == list(range(18))
    assert len(flattened_indices) == len(set(flattened_indices))
    assert len(decoder.stages) == 3
    assert decoder.stages[0].context_attention is None
    assert decoder.stages[0].context_norm is None
    assert isinstance(decoder.stages[1].context_attention, nn.MultiheadAttention)
    assert isinstance(decoder.stages[2].context_attention, nn.MultiheadAttention)
    assert decoder.adjacency.shape == (18, 18)
    assert "adjacency" in dict(decoder.named_buffers())
    assert "adjacency" not in dict(decoder.named_parameters())
    assert isinstance(decoder.joint_attention, nn.MultiheadAttention)
    assert decoder.joint_attention.embed_dim == 256
    assert decoder.joint_attention.num_heads == 4
    assert isinstance(decoder.coordinate_head[0], nn.Linear)
    assert decoder.coordinate_head[0].in_features == 256
    assert decoder.coordinate_head[0].out_features == 128
    assert decoder.coordinate_head[2].out_features == 2


def test_wiflow_hierarchical_joint_decoder_flattens_spatial_tokens() -> None:
    decoder = WiFlowHierarchicalJointDecoder()
    x = torch.randn(2, 256, 29, 16)

    tokens = decoder.flatten_tokens(x)

    assert tokens.shape == (2, 464, 256)