from __future__ import annotations

import torch

from models import WiFlowSpatialTemporalFuser


def test_wiflow_spatial_temporal_fuser_output_shape() -> None:
    fuser = WiFlowSpatialTemporalFuser(sequence_length=8)
    x = torch.randn(2, 8, 256, 29, 10)

    y = fuser(x)

    assert y.shape == (2, 256, 29, 10)


def test_wiflow_spatial_temporal_fuser_position_embedding_shape() -> None:
    fuser = WiFlowSpatialTemporalFuser(sequence_length=8)

    assert fuser.sequence_length == 8
    assert fuser.embedding_dim == 256
    assert fuser.position_embedding.shape == (1, 8, 256)
    assert torch.equal(fuser.position_embedding, torch.zeros_like(fuser.position_embedding))


def test_wiflow_spatial_temporal_fuser_extracts_middle_feature_map() -> None:
    fuser = WiFlowSpatialTemporalFuser(sequence_length=8)
    fuser.eval()
    with torch.no_grad():
        for parameter in fuser.self_attention.parameters():
            parameter.zero_()

    x = torch.randn(2, 8, 256, 29, 10)

    y = fuser(x)

    assert fuser.middle_index == 4
    expected = fuser.output_norm(x[:, 4].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    assert torch.allclose(y, expected)
