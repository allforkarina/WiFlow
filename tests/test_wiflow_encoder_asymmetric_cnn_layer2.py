from __future__ import annotations

import torch

from models import WiFlowEncoderAsymmetricCNNLayer2


def test_wiflow_encoder_asymmetric_cnn_layer2_output_shape() -> None:
    layer = WiFlowEncoderAsymmetricCNNLayer2()
    x = torch.randn(4, 3, 114, 10)

    y = layer(x)

    assert y.shape == (4, 64, 10, 29)


def test_wiflow_encoder_asymmetric_cnn_layer2_stage_shapes() -> None:
    layer = WiFlowEncoderAsymmetricCNNLayer2()
    x = torch.randn(2, 3, 114, 10)

    reshaped = layer._reshape_input(x)
    stem_output = layer.stem(reshaped)
    resblock1_output = layer.resblock1(stem_output)
    resblock2_output = layer.resblock2(resblock1_output)
    resblock3_output = layer.resblock3(resblock2_output)
    assert reshaped.shape == (2, 3, 10, 114)
    assert stem_output.shape == (2, 16, 10, 114)
    assert resblock1_output.shape == (2, 32, 10, 57)
    assert resblock2_output.shape == (2, 64, 10, 29)
    assert resblock3_output.shape == (2, 64, 10, 29)


def test_wiflow_encoder_asymmetric_cnn_layer2_shortcuts_match_main_path() -> None:
    layer = WiFlowEncoderAsymmetricCNNLayer2()
    x = torch.randn(2, 3, 114, 10)

    stem_output = layer.stem(layer._reshape_input(x))
    resblock1_main = layer.resblock1.main_path(stem_output)
    resblock1_shortcut = layer.resblock1.shortcut(stem_output)
    resblock2_main = layer.resblock2.main_path(resblock1_main)
    resblock2_shortcut = layer.resblock2.shortcut(resblock1_main)
    resblock3_main = layer.resblock3.main_path(resblock2_main)
    resblock3_shortcut = layer.resblock3.shortcut(resblock2_main)

    assert resblock1_main.shape == resblock1_shortcut.shape == (2, 32, 10, 57)
    assert resblock2_main.shape == resblock2_shortcut.shape == (2, 64, 10, 29)
    assert resblock3_main.shape == resblock3_shortcut.shape == (2, 64, 10, 29)


def test_wiflow_encoder_asymmetric_cnn_layer2_supports_single_item_batch() -> None:
    layer = WiFlowEncoderAsymmetricCNNLayer2()
    x = torch.randn(1, 3, 114, 10)

    y = layer(x)

    assert y.shape == (1, 64, 10, 29)
