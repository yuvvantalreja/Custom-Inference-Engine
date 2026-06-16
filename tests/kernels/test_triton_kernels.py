"""Triton kernel correctness tests vs reference."""

import pytest
import torch

from kernels.ref import flash_attention_ref, int8_matmul_ref

triton = pytest.importorskip("triton")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flash_attention_matches_reference():
    from kernels.flash_attention import flash_attention

    torch.manual_seed(0)
    B, H, Lq, Lk, D = 2, 4, 64, 64, 64
    q = torch.randn(B, H, Lq, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, Lk, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, Lk, D, device="cuda", dtype=torch.float16)
    out = flash_attention(q, k, v, causal=True)
    ref = flash_attention_ref(q, k, v, causal=True)
    assert (out - ref).abs().max() < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_int8_matmul_matches_reference():
    from kernels.int8_matmul import int8_matmul

    torch.manual_seed(0)
    M, N, K = 64, 128, 64
    w_fp = torch.randn(N, K)
    scales = (w_fp.abs().amax(dim=1) / 127.0).clamp_min(1e-8)
    w_int8 = (w_fp / scales.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)

    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w_int8 = w_int8.cuda()
    scales_cu = scales.cuda().to(torch.float16)

    y = int8_matmul(x, w_int8, scales_cu)
    ref = int8_matmul_ref(x, w_int8, scales_cu)
    assert (y - ref).abs().max() < 1e-3
