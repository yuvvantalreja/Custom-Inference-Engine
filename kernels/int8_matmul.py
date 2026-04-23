"""Triton INT8 x FP16 matmul with dequantize-on-load.

Performs ``y = x @ dq(w_int8, scales).T`` where weights are INT8 per-channel
scales. Accumulation is FP32; output is in x.dtype.
"""

from __future__ import annotations

import torch

from kernels import triton_available
from kernels.ref import int8_matmul_ref


if triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _int8_matmul_fwd(
        X, W, S, Y,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + offs_k
            x_ptrs = X + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk
            w_ptrs = W + offs_n[:, None] * stride_wn + k_idx[None, :] * stride_wk
            mask_k = k_idx < K
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0).to(tl.float32)
            w_q = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & mask_k[None, :], other=0).to(tl.float32)
            acc += tl.dot(x, tl.trans(w_q))

        s = tl.load(S + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        y = acc * s[None, :]
        y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, y.to(Y.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def int8_matmul(
    x: torch.Tensor,
    w_int8: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-channel INT8 x FP16 matmul. Falls back to reference on CPU."""
    if not (x.is_cuda and triton_available() and scales.ndim == 1):
        return int8_matmul_ref(x, w_int8, scales, bias=bias)

    x_2d = x.reshape(-1, x.shape[-1])
    M, K = x_2d.shape
    N, K2 = w_int8.shape
    assert K == K2

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _int8_matmul_fwd[grid](
        x_2d, w_int8, scales, y,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        w_int8.stride(0), w_int8.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    y = y.reshape(*x.shape[:-1], N)
    if bias is not None:
        y = y + bias.to(y.dtype)
    return y
