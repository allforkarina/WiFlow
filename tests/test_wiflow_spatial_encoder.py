from __future__ import annotations

import torch

from models import WiFlowSpatialEncoder


def test_wiflow_spatial_encoder_output_shape_default_features() -> None:
    layer = WiFlowSpatialEncoder(input_channels=6)
    x = torch.randn(4, 6, 114, 10)

    y = layer(x)

    assert y.shape == (4, 128, 29, 10)


def test_wiflow_spatial_encoder_supports_amp_only() -> None:
    layer = WiFlowSpatialEncoder(input_channels=3)
    x = torch.randn(1, 3, 114, 10)

    y = layer(x)

    assert y.shape == (1, 128, 29, 10)


def test_wiflow_spatial_encoder_stage_shapes() -> None:
    layer = WiFlowSpatialEncoder(input_channels=6)
    x = torch.randn(2, 6, 114, 10)

    conv_input = layer._to_conv_layout(x)
    stem_output = layer.stem(conv_input)
    resblock1_output = layer.resblock1(stem_output)
    resblock2_output = layer.resblock2(resblock1_output)
    resblock3_output = layer.resblock3(resblock2_output)

    assert conv_input.shape == (2, 6, 10, 114)
    assert stem_output.shape == (2, 32, 10, 114)
    assert resblock1_output.shape == (2, 64, 10, 57)
    assert resblock2_output.shape == (2, 128, 10, 29)
    assert resblock3_output.shape == (2, 128, 10, 29)
    assert layer._to_model_layout(resblock3_output).shape == (2, 128, 29, 10)
