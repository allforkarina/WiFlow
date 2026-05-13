from __future__ import annotations

import torch

from models import DECODER_TYPES, WiFlowHierarchicalJointDecoder, WiFlowJointDecoder, WiFlowMSFNDecoder, WiFlowModel
from eval import load_checkpoint_model


def test_load_checkpoint_model_uses_train_config(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pth"
    model = WiFlowModel(input_channels=3, axial_mode="parallel_sum")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_config": {
                "axial_mode": "parallel_sum",
                "input_channels": 3,
            },
        },
        checkpoint_path,
    )

    loaded_model, input_channels = load_checkpoint_model(checkpoint_path, torch.device("cpu"))

    assert isinstance(loaded_model, WiFlowModel)
    assert loaded_model.input_channels == 3
    assert loaded_model.axial_mode == "parallel_sum"
    assert loaded_model.decoder_type == "joint"
    assert isinstance(loaded_model.decoder, WiFlowJointDecoder)
    assert input_channels == 3


def test_load_checkpoint_model_uses_decoder_type(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pth"
    model = WiFlowModel(input_channels=3, decoder_type="hierarchical")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_config": {
                "decoder_type": "hierarchical",
                "input_channels": 3,
            },
        },
        checkpoint_path,
    )

    loaded_model, _ = load_checkpoint_model(checkpoint_path, torch.device("cpu"))

    assert loaded_model.decoder_type == "hierarchical"
    assert isinstance(loaded_model.decoder, WiFlowHierarchicalJointDecoder)


def test_load_checkpoint_model_uses_heatmap_decoder_type(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pth"
    model = WiFlowModel(input_channels=3, decoder_type="heatmap_msfn", heatmap_size=40)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_config": {
                "decoder_type": "heatmap_msfn",
                "heatmap_size": 40,
                "input_channels": 3,
            },
        },
        checkpoint_path,
    )

    loaded_model, _ = load_checkpoint_model(checkpoint_path, torch.device("cpu"))

    assert loaded_model.decoder_type == "heatmap_msfn"
    assert loaded_model.heatmap_size == 40
    assert isinstance(loaded_model.decoder, WiFlowMSFNDecoder)


def test_load_checkpoint_model_requires_model_state_dict(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save({"train_config": {"input_channels": 3}}, checkpoint_path)

    try:
        load_checkpoint_model(checkpoint_path, torch.device("cpu"))
    except KeyError as exc:
        assert "model_state_dict" in str(exc)
    else:
        raise AssertionError("Expected checkpoint loading to require model_state_dict")


def test_load_checkpoint_model_requires_train_config(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": WiFlowModel().state_dict()}, checkpoint_path)

    try:
        load_checkpoint_model(checkpoint_path, torch.device("cpu"))
    except KeyError as exc:
        assert "train_config" in str(exc)
    else:
        raise AssertionError("Expected checkpoint loading to require train_config")