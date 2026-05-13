from __future__ import annotations

import torch

from models import (
    DECODER_TYPES,
    WiFlowAxialEncoder,
    WiFlowMSFNDecoder,
    WiFlowHierarchicalJointDecoder,
    WiFlowJointDecoder,
    WiFlowModel,
    WiFlowSpatialEncoder,
)


def test_wiflow_model_output_shape_default_features() -> None:
    model = WiFlowModel()
    x = torch.randn(4, 3, 114, 64)

    y = model(x)

    assert y.shape == (4, 18, 2)
    assert model.decoder_type == "joint"


def test_wiflow_model_supports_amp_only() -> None:
    model = WiFlowModel(input_channels=3)
    x = torch.randn(1, 3, 114, 64)

    y = model(x)

    assert y.shape == (1, 18, 2)


def test_wiflow_model_supports_axial_mode() -> None:
    model = WiFlowModel(axial_mode="temporal_then_spatial")
    x = torch.randn(1, 3, 114, 64)

    y = model(x)

    assert model.axial_mode == "temporal_then_spatial"
    assert model.axial_encoder.mode == "temporal_then_spatial"
    assert y.shape == (1, 18, 2)


def test_wiflow_model_uses_expected_modules() -> None:
    model = WiFlowModel()

    assert isinstance(model.spatial_encoder, WiFlowSpatialEncoder)
    assert isinstance(model.axial_encoder, WiFlowAxialEncoder)
    assert not hasattr(model, "pooler")
    assert isinstance(model.decoder, WiFlowJointDecoder)


def test_wiflow_model_supports_hierarchical_decoder() -> None:
    model = WiFlowModel(decoder_type="hierarchical")
    x = torch.randn(2, 3, 114, 64)

    y = model(x)

    assert model.decoder_type == "hierarchical"
    assert isinstance(model.decoder, WiFlowHierarchicalJointDecoder)
    assert y.shape == (2, 18, 2)


def test_wiflow_model_supports_heatmap_msfn_decoder() -> None:
    model = WiFlowModel(decoder_type="heatmap_msfn")
    x = torch.randn(2, 3, 114, 64)

    y = model(x)

    assert model.decoder_type == "heatmap_msfn"
    assert isinstance(model.decoder, WiFlowMSFNDecoder)
    assert y["keypoints"].shape == (2, 18, 2)
    assert len(y["stages"]) == 3
    assert y["stages"][-1]["pcm"].shape == (2, 18, 36, 36)
    assert y["stages"][-1]["paf"].shape == (2, 38, 36, 36)


def test_wiflow_model_rejects_unknown_decoder_type() -> None:
    assert DECODER_TYPES == ("joint", "hierarchical", "heatmap_msfn")
    try:
        WiFlowModel(decoder_type="unknown")
    except ValueError as exc:
        assert "decoder_type" in str(exc)
    else:
        raise AssertionError("Expected WiFlowModel to reject unknown decoder_type")


def test_wiflow_model_rejects_5d_input() -> None:
    model = WiFlowModel()
    x = torch.randn(2, 8, 3, 114, 64)

    try:
        model(x)
    except ValueError as exc:
        assert "expects input shaped" in str(exc)
    else:
        raise AssertionError("Expected WiFlowModel to reject 5D input")