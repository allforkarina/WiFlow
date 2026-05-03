from __future__ import annotations

import torch

from models import WiFlowTemporalEncoder


def test_wiflow_temporal_encoder_output_shape() -> None:
    encoder = WiFlowTemporalEncoder(sequence_length=8)
    x = torch.randn(2, 8, 256)

    y = encoder(x)

    assert y.shape == (2, 256)


def test_wiflow_temporal_encoder_position_embedding_shape() -> None:
    encoder = WiFlowTemporalEncoder(sequence_length=8)

    assert encoder.position_embedding.shape == (1, 8, 256)
    assert torch.equal(encoder.position_embedding, torch.zeros_like(encoder.position_embedding))


def test_wiflow_temporal_encoder_extracts_middle_token() -> None:
    encoder = WiFlowTemporalEncoder(sequence_length=8)
    encoder.eval()
    with torch.no_grad():
        for parameter in encoder.self_attention.parameters():
            parameter.zero_()

    x = torch.randn(2, 8, 256)

    y = encoder(x)

    assert encoder.middle_index == 4
    assert torch.allclose(y, encoder.output_norm(x[:, 4]))
