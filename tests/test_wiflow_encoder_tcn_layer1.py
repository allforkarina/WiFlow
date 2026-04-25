from __future__ import annotations

import torch

from models import WiFlowEncoderTCNLayer1


def test_wiflow_encoder_tcn_layer1_preserves_shape() -> None:
    layer = WiFlowEncoderTCNLayer1()
    x = torch.randn(4, 342, 10)

    y = layer(x)

    assert y.shape == (4, 340, 10)


def test_wiflow_encoder_tcn_layer1_is_causal() -> None:
    torch.manual_seed(0)
    layer = WiFlowEncoderTCNLayer1()
    layer.eval()
    x_past = torch.randn(2, 342, 10)
    x_future_changed = x_past.clone()
    x_future_changed[:, :, 6:] = torch.randn(2, 342, 4)

    y_past = layer(x_past)
    y_future_changed = layer(x_future_changed)

    assert torch.allclose(y_past[:, :, :6], y_future_changed[:, :, :6])


def test_wiflow_encoder_tcn_layer1_supports_single_item_batch() -> None:
    layer = WiFlowEncoderTCNLayer1()
    x = torch.randn(1, 342, 10)

    y = layer(x)

    assert y.shape == (1, 340, 10)


def test_wiflow_encoder_tcn_layer1_configuration() -> None:
    layer = WiFlowEncoderTCNLayer1()

    assert layer.in_channels == 342
    assert layer.temporal_channels == 342
    assert layer.out_channels == 340
    assert layer.kernel_size == 3
    assert layer.dilations == (1, 2, 4, 8)
    assert len(layer.blocks) == 4
    assert [block.dilation for block in layer.blocks] == [1, 2, 4, 8]
    assert layer.blocks[-1].temporal_conv.in_channels == 342
    assert layer.blocks[-1].temporal_conv.out_channels == 340
