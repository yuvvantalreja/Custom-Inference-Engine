"""Triton Flash Attention kernel for sm_75 (T4).

Uses the FP16 tensor-core path: Q/K/V stay in FP16 for the matmuls, the online
softmax accumulator is FP32. Tile sizes are chosen for T4's 64 KiB shared
memory — large head dims use narrower N-tiles. ``num_stages=1`` because sm_75
has no ``cp.async`` (software pipelining would either be ignored or double
the smem footprint).
"""

from __future__ import annotations

import math

import torch

from kernels import triton_available
from kernels.ref import flash_attention_ref


if triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _flash_attn_fwd(
        Q, K, V, Out,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        B, H, M, N, D: tl.constexpr,
        scale,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        offs_n = tl.arange(0, BLOCK_N)

        q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        # Keep Q in FP16 for the tensor-core path; tl.dot accumulates to FP32.
        q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)

        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

        n_end = N
        if IS_CAUSAL:
            n_end = tl.minimum(N, (pid_m + 1) * BLOCK_M + (N - M))

        for start_n in range(0, n_end, BLOCK_N):
            k_ptrs = K + b * stride_kb + h * stride_kh + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = V + b * stride_vb + h * stride_vh + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vd
            mask_n = (start_n + offs_n) < N
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)  # FP16
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)  # FP16

            # FP16 x FP16 -> FP32 accumulator on Turing tensor cores.
            s = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
            if IS_CAUSAL:
                i = offs_m[:, None]
                j = (start_n + offs_n)[None, :]
                s = tl.where(j > (N - M) + i, float("-inf"), s)
            s = tl.where(mask_n[None, :], s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            # Rescale running acc in FP32, then add FP16xFP16 -> FP32 product.
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
            m_i = m_new

        acc = acc / l_i[:, None]
        out_ptrs = Out + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < M)


def _tile_for_t4(M: int, D: int) -> tuple[int, int]:
    """Pick (BLOCK_M, BLOCK_N) safe for 64 KiB smem on sm_75.

    Budget in FP16: Q+K+V+acc fit comfortably when BLOCK_M*D + 2*BLOCK_N*D +
    BLOCK_M*D*2 (acc is FP32) stays under ~48 KiB.
    """
    if D >= 128:
        bm = 64 if M >= 64 else 32
        bn = 32
    elif D >= 64:
        bm = 64 if M >= 64 else 32
        bn = 64
    else:
        bm = 64 if M >= 64 else 16
        bn = 64
    return bm, bn


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Flash attention. Falls back to reference on CPU / when Triton is unavailable.

    q: (B, H, M, D), k/v: (B, Hk, N, D). If Hk != H, keys/values are repeat-interleaved.
    """
    if not (q.is_cuda and triton_available()):
        return flash_attention_ref(q, k, v, causal=causal, scale=scale)

    B, H, M, D = q.shape
    _, Hk, N, _ = k.shape
    if Hk != H:
        repeat = H // Hk
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    assert (D & (D - 1)) == 0, "head_dim must be a power of 2 for the Triton kernel"
    assert q.dtype == torch.float16, f"Triton kernel expects FP16 Q/K/V, got {q.dtype}"
    BLOCK_M, BLOCK_N = _tile_for_t4(M, D)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)
    grid = (triton.cdiv(M, BLOCK_M), B * H)
    _flash_attn_fwd[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, M, N, D,
        scale,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=1,
    )
    return out
