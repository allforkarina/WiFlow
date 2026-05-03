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
    assert model.sequence_length == 1
    assert model.temporal_encoder is None


def test_wiflow_model_sequence_length_one_uses_single_frame_input() -> None:
    model = WiFlowModel(sequence_length=1)
    x = torch.randn(2, 6, 114, 10)

    y = model(x)

    assert y.shape == (2, 17, 2)


def test_wiflow_model_supports_amp_only() -> None:
    model = WiFlowModel(input_channels=3)
    x = torch.randn(1, 3, 114, 10)

    y = model(x)

    assert y.shape == (1, 17, 2)


def test_wiflow_model_supports_axial_mode() -> None:
    model = WiFlowModel(axial_mode="temporal_then_spatial")
    x = torch.randn(1, 6, 114, 10)

    y = model(x)

    assert model.axial_mode == "temporal_then_spatial"
    assert model.axial_encoder.mode == "temporal_then_spatial"
    assert y.shape == (1, 17, 2)


def test_wiflow_model_supports_temporal_sequence_input() -> None:
    model = WiFlowModel(input_channels=6, sequence_length=8)
    x = torch.randn(2, 8, 6, 114, 10)

    y = model(x)

    assert model.sequence_length == 8
    assert model.temporal_encoder is not None
    assert y.shape == (2, 17, 2)


def test_wiflow_model_uses_expected_modules() -> None:
    model = WiFlowModel()

    assert isinstance(model.spatial_encoder, WiFlowSpatialEncoder)
    assert isinstance(model.axial_encoder, WiFlowAxialEncoder)
    assert isinstance(model.pooler, WiFlowAttentionPooler)
    assert isinstance(model.decoder, WiFlowSkeletonDecoder)
