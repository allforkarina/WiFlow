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
    assert len(layer.antenna_mixers) == 1
    assert len(layer.feature_stems) == 1


def test_wiflow_spatial_encoder_supports_three_feature_groups() -> None:
    layer = WiFlowSpatialEncoder(input_channels=9)
    x = torch.randn(1, 9, 114, 10)

    y = layer(x)

    assert y.shape == (1, 128, 29, 10)
    assert len(layer.antenna_mixers) == 3
    assert len(layer.feature_stems) == 3
    assert [stem[0].out_channels for stem in layer.feature_stems] == [11, 11, 10]


def test_wiflow_spatial_encoder_requires_three_antenna_feature_groups() -> None:
    for input_channels in (0, 5):
        try:
            WiFlowSpatialEncoder(input_channels=input_channels)
        except ValueError as exc:
            assert "positive multiple of 3" in str(exc)
        else:
            raise AssertionError("Expected WiFlowSpatialEncoder to reject invalid input channels")


def test_wiflow_spatial_encoder_stage_shapes() -> None:
    layer = WiFlowSpatialEncoder(input_channels=6)
    x = torch.randn(2, 6, 114, 10)

    conv_input = layer._to_conv_layout(x)
    stem_output = layer._apply_feature_stems(conv_input)
    resblock1_output = layer.resblock1(stem_output)
    resblock2_output = layer.resblock2(resblock1_output)
    resblock3_output = layer.resblock3(resblock2_output)

    assert conv_input.shape == (2, 6, 10, 114)
    assert len(layer.antenna_mixers) == 2
    assert len(layer.feature_stems) == 2
    assert stem_output.shape == (2, 32, 10, 114)
    assert resblock1_output.shape == (2, 64, 10, 57)
    assert resblock2_output.shape == (2, 128, 10, 29)
    assert resblock3_output.shape == (2, 128, 10, 29)
    assert layer._to_model_layout(resblock3_output).shape == (2, 128, 29, 10)


def test_wiflow_spatial_encoder_uses_time_frequency_resblocks() -> None:
    layer = WiFlowSpatialEncoder(input_channels=6)

    first_conv = layer.resblock1.main_path[0]
    second_conv = layer.resblock1.main_path[3]

    assert first_conv.kernel_size == (3, 3)
    assert first_conv.padding == (1, 1)
    assert first_conv.stride == (1, 2)
    assert second_conv.kernel_size == (3, 3)
    assert second_conv.padding == (1, 1)
    assert second_conv.stride == (1, 1)
