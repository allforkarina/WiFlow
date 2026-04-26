from __future__ import annotations

import torch
from torch import nn

from models import TemporalAttentionPooling, WiFlowDecoder


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
    assert isinstance(decoder.temporal_pool, TemporalAttentionPooling)
    assert isinstance(decoder.temporal_pool.attention_logits, nn.Conv2d)
    assert decoder.temporal_pool.attention_logits.in_channels == 32
    assert decoder.temporal_pool.attention_logits.out_channels == 1
    assert decoder.temporal_pool.attention_logits.kernel_size == (1, 1)
    assert decoder.joint_embedding.shape == (17, 32)
    assert isinstance(decoder.x_head[0], nn.Linear)
    assert decoder.x_head[0].in_features == 32
    assert decoder.x_head[0].out_features == 32
    assert isinstance(decoder.x_head[1], nn.SiLU)
    assert isinstance(decoder.x_head[2], nn.Linear)
    assert decoder.x_head[2].in_features == 32
    assert decoder.x_head[2].out_features == 128
    assert isinstance(decoder.y_head[0], nn.Linear)
    assert decoder.y_head[2].out_features == 128
    assert decoder.x_bin_centers.shape == (128,)
    assert decoder.y_bin_centers.shape == (128,)


def test_temporal_attention_pooling_weights_sum_to_one() -> None:
    pool = TemporalAttentionPooling(channels=32)
    x = torch.randn(2, 32, 17, 10)

    weights = pool.compute_attention_weights(x)
    y = pool(x)

    assert weights.shape == (2, 1, 17, 10)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 1, 17))
    assert y.shape == (2, 32, 17, 1)


def test_wiflow_decoder_uses_no_adaptive_average_pooling() -> None:
    decoder = WiFlowDecoder()

    assert not any(isinstance(module, nn.AdaptiveAvgPool2d) for module in decoder.modules())


def test_wiflow_decoder_uses_joint_aware_linear_head() -> None:
    decoder = WiFlowDecoder()

    assert any(isinstance(module, nn.Linear) for module in decoder.x_head)
    assert any(isinstance(module, nn.Linear) for module in decoder.y_head)


def test_wiflow_decoder_forward_with_logits_returns_distribution_shapes() -> None:
    decoder = WiFlowDecoder()
    x = torch.randn(2, 64, 17, 10)

    prediction, x_logits, y_logits = decoder.forward_with_logits(x)

    assert prediction.shape == (2, 17, 2)
    assert x_logits.shape == (2, 17, 128)
    assert y_logits.shape == (2, 17, 128)


def test_decode_coordinate_distribution_matches_peaked_logits() -> None:
    decoder = WiFlowDecoder(num_x_bins=8, num_y_bins=8)
    x_logits = torch.full((1, 17, 8), -20.0)
    y_logits = torch.full((1, 17, 8), -20.0)
    x_logits[..., 3] = 20.0
    y_logits[..., 5] = 20.0

    prediction = decoder.decode_coordinate_distribution(x_logits, y_logits)

    assert torch.allclose(prediction[..., 0], torch.full((1, 17), 3.0 / 7.0), atol=1e-4)
    assert torch.allclose(prediction[..., 1], torch.full((1, 17), 5.0 / 7.0), atol=1e-4)
