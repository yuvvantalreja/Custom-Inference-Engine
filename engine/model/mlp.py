from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.quant import Int8Linear


class SwiGLU(nn.Module):
    """SwiGLU MLP: y = W_down (silu(W_gate x) * W_up x)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate = Int8Linear(hidden_size, intermediate_size, bias=False)
        self.up = Int8Linear(hidden_size, intermediate_size, bias=False)
        self.down = Int8Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
