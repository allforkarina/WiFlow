from __future__ import annotations

import torch

from models import WiFlowDecoder, WiFlowEncoder, WiFlowModel


def test_wiflow_model_output_shape() -> None:
    model = WiFlowModel()
    x = torch.randn(4, 342, 10)

    y = model(x)

    assert y.shape == (4, 17, 2)


def test_wiflow_model_supports_single_item_batch() -> None:
    model = WiFlowModel()
    x = torch.randn(1, 342, 10)

    y = model(x)

    assert y.shape == (1, 17, 2)


def test_wiflow_model_uses_expected_modules() -> None:
    model = WiFlowModel()

    assert isinstance(model.encoder, WiFlowEncoder)
    assert isinstance(model.decoder, WiFlowDecoder)
