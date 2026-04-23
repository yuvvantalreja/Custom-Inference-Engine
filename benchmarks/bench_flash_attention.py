from __future__ import annotations

import time

import torch

from kernels.ref import flash_attention_ref


def run(shapes: list[tuple[int, int, int, int, int]] | None = None, device: str = "auto") -> None:
    if shapes is None:
        shapes = [(1, 8, 128, 128, 64), (1, 16, 512, 512, 64), (1, 16, 512, 512, 128)]
    use_cuda = (device == "cuda") or (device == "auto" and torch.cuda.is_available())
    if use_cuda:
        from kernels.flash_attention import flash_attention
    dev = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float16 if use_cuda else torch.float32

    for B, H, Lq, Lk, D in shapes:
        q = torch.randn(B, H, Lq, D, device=dev, dtype=dtype)
        k = torch.randn(B, H, Lk, D, device=dev, dtype=dtype)
        v = torch.randn(B, H, Lk, D, device=dev, dtype=dtype)
        fn = flash_attention if use_cuda else flash_attention_ref
        fn(q, k, v, causal=True)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            fn(q, k, v, causal=True)
        if use_cuda:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 5
        print(f"flash_attn B={B} H={H} Lq={Lq} Lk={Lk} D={D}: {dt*1000:.2f} ms")
