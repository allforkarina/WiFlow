from __future__ import annotations

import torch
from torch import nn

from models import WiFlowDecoder


def test_wiflow_decoder_output_shape() -> None:
    decoder = WiFlowDecoder()
    x = torch.randn(4, 64, 17, 10)

    y = decoder(x)

    assert y.shape == (4, 17, 2)


def test_wiflow_decoder_supports_single_item_batch() -> None:
    decoder = WiFlowDecoder()
    x = torch.randn(1, 64, 17, 10)

    y = decoder(x)

    assert y.shape == (1, 17, 2)


def test_wiflow_decoder_configuration() -> None:
    decoder = WiFlowDecoder()

    assert isinstance(decoder.refinement[0], nn.Conv2d)
    assert decoder.refinement[0].in_channels == 64
    assert decoder.refinement[0].out_channels == 32
    assert decoder.refinement[0].kernel_size == (3, 3)
    assert isinstance(decoder.refinement[1], nn.BatchNorm2d)
    assert isinstance(decoder.refinement[2], nn.SiLU)
    assert isinstance(decoder.coordinate_projection, nn.Conv2d)
    assert decoder.coordinate_projection.in_channels == 32
    assert decoder.coordinate_projection.out_channels == 2
    assert decoder.coordinate_projection.kernel_size == (1, 1)
    assert isinstance(decoder.temporal_pool, nn.AdaptiveAvgPool2d)
    assert decoder.temporal_pool.output_size == (17, 1)


def test_wiflow_decoder_uses_no_linear_layer() -> None:
    decoder = WiFlowDecoder()

    assert not any(isinstance(module, nn.Linear) for module in decoder.modules())
