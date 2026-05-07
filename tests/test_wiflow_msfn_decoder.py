from __future__ import annotations

import torch

from models import WiFlowMSFNDecoder


def test_wiflow_msfn_decoder_outputs_multistage_heatmaps() -> None:
    decoder = WiFlowMSFNDecoder(heatmap_size=36)
    x = torch.randn(2, 256, 29, 10)

    stages = decoder(x)

    assert len(stages) == 3
    for stage in stages:
        assert stage["pcm"].shape == (2, 17, 36, 36)
        assert stage["paf"].shape == (2, 32, 36, 36)


def test_wiflow_msfn_decoder_configuration() -> None:
    decoder = WiFlowMSFNDecoder()

    assert decoder.input_channels == 256
    assert decoder.feature_channels == 128
    assert decoder.hidden_channels == 512
    assert decoder.stages == 3
    assert decoder.heatmap_size == 36
    assert len(decoder.decoders) == 3
    assert len(decoder.papms) == 2
