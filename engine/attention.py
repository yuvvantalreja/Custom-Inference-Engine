from __future__ import annotations

import torch

from kernels.ref import flash_attention_ref

try:
    from kernels.flash_attention import flash_attention as _flash_attention_triton
except Exception:
    _flash_attention_triton = None


def _triton_eligible(q: torch.Tensor) -> bool:
    if not (q.is_cuda and _flash_attention_triton is not None):
        return False
    D = q.shape[-1]
    if D & (D - 1):  # non-power-of-2 head_dim
        return False
    if D < 16:  # Triton tl.dot requires K >= 16 on sm_75
        return False
    if q.dtype != torch.float16:
        return False
    return True


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Dispatch to Triton on CUDA with FP16 + power-of-2 head_dim; reference otherwise.

    Shapes: q (B, H, M, D); k/v (B, Hk, N, D). Supports GQA.
    """
    if _triton_eligible(q):
        return _flash_attention_triton(
            q.contiguous(), k.contiguous(), v.contiguous(),
            causal=causal, scale=scale,
        )
    return flash_attention_ref(q, k, v, causal=causal, scale=scale)
