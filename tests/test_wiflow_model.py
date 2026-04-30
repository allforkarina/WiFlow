from __future__ import annotations

import torch

from models import (
    WiFlowAttentionPooler,
    WiFlowAxialEncoder,
    WiFlowModel,
    WiFlowSkeletonDecoder,
    WiFlowSpatialEncoder,
)


def test_wiflow_model_output_shape_default_features() -> None:
    model = WiFlowModel()
    x = torch.randn(4, 6, 114, 10)

    y = model(x)

    assert y.shape == (4, 17, 2)


def test_wiflow_model_supports_amp_only() -> None:
    model = WiFlowModel(input_channels=3)
    x = torch.randn(1, 3, 114, 10)

    y = model(x)

    assert y.shape == (1, 17, 2)


def test_wiflow_model_uses_expected_modules() -> None:
    model = WiFlowModel()

    assert isinstance(model.spatial_encoder, WiFlowSpatialEncoder)
    assert isinstance(model.axial_encoder, WiFlowAxialEncoder)
    assert isinstance(model.pooler, WiFlowAttentionPooler)
    assert isinstance(model.decoder, WiFlowSkeletonDecoder)
