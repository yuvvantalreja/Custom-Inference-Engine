from __future__ import annotations

import time

import torch

from engine.device import require_cuda
from kernels.flash_attention import flash_attention


def run(shapes: list[tuple[int, int, int, int, int]] | None = None) -> None:
    if shapes is None:
        shapes = [(1, 8, 128, 128, 64), (1, 16, 512, 512, 64), (1, 16, 512, 512, 128)]
    dev = require_cuda()
    dtype = torch.float16

    for B, H, Lq, Lk, D in shapes:
        q = torch.randn(B, H, Lq, D, device=dev, dtype=dtype)
        k = torch.randn(B, H, Lk, D, device=dev, dtype=dtype)
        v = torch.randn(B, H, Lk, D, device=dev, dtype=dtype)
        flash_attention(q, k, v, causal=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            flash_attention(q, k, v, causal=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 5
        print(f"flash_attn B={B} H={H} Lq={Lq} Lk={Lk} D={D}: {dt*1000:.2f} ms")
