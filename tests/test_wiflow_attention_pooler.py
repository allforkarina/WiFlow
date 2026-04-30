from __future__ import annotations

import torch
from torch import nn

from models import WiFlowAttentionPooler


def test_wiflow_attention_pooler_output_shape() -> None:
    pooler = WiFlowAttentionPooler()
    x = torch.randn(4, 256, 29, 10)

    y = pooler(x)

    assert y.shape == (4, 256)


def test_wiflow_attention_pooler_configuration() -> None:
    pooler = WiFlowAttentionPooler()

    assert pooler.global_query.shape == (1, 1, 256)
    assert pooler.global_query.requires_grad
    assert isinstance(pooler.cross_attention, nn.MultiheadAttention)
    assert pooler.cross_attention.embed_dim == 256
    assert pooler.cross_attention.num_heads == 8
    assert pooler.flatten_tokens(torch.randn(2, 256, 29, 10)).shape == (2, 290, 256)
