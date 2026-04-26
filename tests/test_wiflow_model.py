from __future__ import annotations

import torch

from models import WiFlowDecoder, WiFlowEncoder, WiFlowModel


def test_wiflow_model_output_shape() -> None:
    model = WiFlowModel()
    x = torch.randn(4, 3, 114, 10)

    y = model(x)

    assert y.shape == (4, 17, 2)


def test_wiflow_model_supports_single_item_batch() -> None:
    model = WiFlowModel()
    x = torch.randn(1, 3, 114, 10)

    y = model(x)

    assert y.shape == (1, 17, 2)


def test_wiflow_model_forward_with_logits_shapes() -> None:
    model = WiFlowModel()
    x = torch.randn(2, 3, 114, 10)

    prediction, x_logits, y_logits = model.forward_with_logits(x)

    assert prediction.shape == (2, 17, 2)
    assert x_logits.shape == (2, 17, 128)
    assert y_logits.shape == (2, 17, 128)


def test_wiflow_model_uses_expected_modules() -> None:
    model = WiFlowModel()

    assert isinstance(model.encoder, WiFlowEncoder)
    assert isinstance(model.decoder, WiFlowDecoder)
