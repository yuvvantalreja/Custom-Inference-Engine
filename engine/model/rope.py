from __future__ import annotations

import torch
import torch.nn as nn

from kernels.ref import build_rope_cache, apply_rotary_ref


class RotaryEmbedding(nn.Module):
    """RoPE with a precomputed cos/sin cache."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        cos, sin = build_rope_cache(max_seq_len, head_dim, theta=theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def apply(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        return apply_rotary_ref(x, self.cos, self.sin, positions=positions)
