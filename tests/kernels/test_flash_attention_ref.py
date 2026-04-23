import math

import pytest
import torch

from kernels.ref import flash_attention_ref


@pytest.mark.parametrize("B,H,Lq,Lk,D", [(1, 2, 4, 4, 8), (2, 4, 8, 16, 16)])
def test_matches_manual_softmax(B, H, Lq, Lk, D):
    torch.manual_seed(0)
    q = torch.randn(B, H, Lq, D)
    k = torch.randn(B, H, Lk, D)
    v = torch.randn(B, H, Lk, D)
    out = flash_attention_ref(q, k, v)
    scores = q @ k.transpose(-2, -1) / math.sqrt(D)
    expected = torch.softmax(scores, dim=-1) @ v
    assert torch.allclose(out, expected, atol=1e-5)


def test_causal_mask_blocks_future():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    full = flash_attention_ref(q, k, v)
    causal = flash_attention_ref(q, k, v, causal=True)
    # first position attends to only itself under causal, so output differs
    assert not torch.allclose(full, causal)


def test_gqa_broadcast():
    q = torch.randn(1, 4, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)
    out = flash_attention_ref(q, k, v)
    assert out.shape == (1, 4, 3, 8)
