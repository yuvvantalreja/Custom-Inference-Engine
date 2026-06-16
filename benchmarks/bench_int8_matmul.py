from __future__ import annotations

import time

import torch

from engine.device import require_cuda
from engine.quant.calibrate import calibrate_weight
from kernels.int8_matmul import int8_matmul


def run(shapes: list[tuple[int, int, int]] | None = None) -> None:
    if shapes is None:
        shapes = [(64, 4096, 4096), (512, 4096, 4096)]
    dev = require_cuda()
    dtype = torch.float16

    for M, N, K in shapes:
        w_fp = torch.randn(N, K)
        w_int8, scales = calibrate_weight(w_fp)
        w_int8 = w_int8.to(dev)
        scales = scales.to(dev).to(dtype)
        x = torch.randn(M, K, device=dev, dtype=dtype)

        int8_matmul(x, w_int8, scales)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            int8_matmul(x, w_int8, scales)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 5
        tflops = 2 * M * N * K / dt / 1e12
        print(f"int8_matmul M={M} N={N} K={K}: {dt*1000:.2f} ms, {tflops:.2f} TFLOP/s")
