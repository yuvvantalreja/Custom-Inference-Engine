from __future__ import annotations

import torch


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Triton flash attention on CUDA.

    Shapes: q (B, H, M, D); k/v (B, Hk, N, D). Supports GQA.
    """
    if not q.is_cuda:
        raise RuntimeError("flash_attention requires CUDA tensors")
    D = q.shape[-1]
    if D & (D - 1):
        raise ValueError(f"head_dim must be power of 2 for Triton kernel, got {D}")
    if D < 16:
        raise ValueError(f"head_dim must be >= 16 for Triton kernel on sm_75, got {D}")
    if q.dtype != torch.float16:
        raise ValueError(f"Triton kernel expects FP16 Q/K/V, got {q.dtype}")

    from kernels.flash_attention import flash_attention as _flash_attention_triton

    return _flash_attention_triton(
        q.contiguous(), k.contiguous(), v.contiguous(),
        causal=causal, scale=scale,
    )
