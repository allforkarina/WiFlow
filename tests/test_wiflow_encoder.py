from __future__ import annotations

import torch

from models import (
    WiFlowEncoder,
    WiFlowEncoderAsymmetricCNNLayer2,
    WiFlowEncoderAxialAttentionLayer3,
)


def test_wiflow_encoder_output_shape() -> None:
    encoder = WiFlowEncoder()
    x = torch.randn(4, 3, 114, 10)

    y = encoder(x)

    assert y.shape == (4, 64, 17, 10)


def test_wiflow_encoder_supports_single_item_batch() -> None:
    encoder = WiFlowEncoder()
    x = torch.randn(1, 3, 114, 10)

    y = encoder(x)

    assert y.shape == (1, 64, 17, 10)


def test_wiflow_encoder_uses_expected_layers() -> None:
    encoder = WiFlowEncoder()

    assert isinstance(encoder.layer1, WiFlowEncoderAsymmetricCNNLayer2)
    assert isinstance(encoder.layer3, WiFlowEncoderAxialAttentionLayer3)


def test_wiflow_encoder_stage_shapes() -> None:
    encoder = WiFlowEncoder()
    x = torch.randn(2, 3, 114, 10)

    layer1_output = encoder.layer1(x)
    axial_input = encoder._prepare_axial_attention_input(layer1_output)
    layer3_output = encoder.layer3(axial_input)

    assert layer1_output.shape == (2, 64, 10, 17)
    assert axial_input.shape == (2, 64, 17, 10)
    assert layer3_output.shape == (2, 64, 17, 10)
