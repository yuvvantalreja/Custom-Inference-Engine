from __future__ import annotations

import math

import torch


def flash_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Naive scaled-dot-product attention in FP32.

    Shapes:
      q: (B, H, Lq, D)
      k: (B, Hk, Lk, D)
      v: (B, Hk, Lk, D)
    If H != Hk, k/v are broadcast across heads (grouped-query attention).
    mask: additive float mask of shape broadcastable to (B, H, Lq, Lk), or None.
    Returns: (B, H, Lq, D) in the input dtype.
    """
    in_dtype = q.dtype
    q32 = q.float()
    k32 = k.float()
    v32 = v.float()

    B, H, Lq, D = q32.shape
    _, Hk, Lk, _ = k32.shape
    if Hk != H:
        assert H % Hk == 0, f"num_heads {H} must be a multiple of num_kv_heads {Hk}"
        repeat = H // Hk
        k32 = k32.repeat_interleave(repeat, dim=1)
        v32 = v32.repeat_interleave(repeat, dim=1)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    scores = torch.matmul(q32, k32.transpose(-2, -1)) * scale  # (B,H,Lq,Lk)

    if causal:
        # positions: query i attends to keys [0, Lk - Lq + i]
        i = torch.arange(Lq, device=q.device).unsqueeze(-1)
        j = torch.arange(Lk, device=q.device).unsqueeze(0)
        causal_mask = j > (Lk - Lq + i)
        scores = scores.masked_fill(causal_mask, float("-inf"))

    if mask is not None:
        scores = scores + mask

    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v32)
    return out.to(in_dtype)
