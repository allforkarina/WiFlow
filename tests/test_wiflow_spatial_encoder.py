from __future__ import annotations

import torch

from models import WiFlowSpatialEncoder


def test_wiflow_spatial_encoder_output_shape_default_features() -> None:
    layer = WiFlowSpatialEncoder(input_channels=3)
    x = torch.randn(4, 3, 114, 64)

    y = layer(x)

    assert y.shape == (4, 128, 29, 16)


def test_wiflow_spatial_encoder_supports_amp_only() -> None:
    layer = WiFlowSpatialEncoder(input_channels=3)
    x = torch.randn(1, 3, 114, 64)

    y = layer(x)

    assert y.shape == (1, 128, 29, 16)
    assert isinstance(layer.antenna_mixer, torch.nn.Sequential)
    assert isinstance(layer.feature_stem, torch.nn.Sequential)


def test_wiflow_spatial_encoder_requires_three_channels() -> None:
    for input_channels in (0, 6, 9):
        try:
            WiFlowSpatialEncoder(input_channels=input_channels)
        except ValueError as exc:
            assert "input_channels must be 3" in str(exc)
        else:
            raise AssertionError("Expected WiFlowSpatialEncoder to reject invalid input channels")


def test_wiflow_spatial_encoder_stage_shapes() -> None:
    layer = WiFlowSpatialEncoder(input_channels=3)
    x = torch.randn(2, 3, 114, 64)

    conv_input = layer._to_conv_layout(x)
    stem_output = layer.feature_stem(layer.antenna_mixer(conv_input))
    resblock1_output = layer.resblock1(stem_output)
    resblock2_output = layer.resblock2(resblock1_output)
    resblock3_output = layer.resblock3(resblock2_output)

    assert conv_input.shape == (2, 3, 64, 114)
    assert stem_output.shape == (2, 32, 64, 114)
    assert resblock1_output.shape == (2, 64, 32, 57)
    assert resblock2_output.shape == (2, 128, 16, 29)
    assert resblock3_output.shape == (2, 128, 16, 29)
    assert layer._to_model_layout(resblock3_output).shape == (2, 128, 29, 16)


def test_wiflow_spatial_encoder_uses_symmetric_time_frequency_resblocks() -> None:
    layer = WiFlowSpatialEncoder(input_channels=3)

    first_conv = layer.resblock1.main_path[0]
    second_conv = layer.resblock1.main_path[3]

    assert first_conv.kernel_size == (3, 3)
    assert first_conv.padding == (1, 1)
    assert first_conv.stride == (2, 2)
    assert second_conv.kernel_size == (3, 3)
    assert second_conv.padding == (1, 1)
    assert second_conv.stride == (1, 1)