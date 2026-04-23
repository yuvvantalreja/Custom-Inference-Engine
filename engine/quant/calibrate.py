from __future__ import annotations

import torch


def calibrate_weight(
    w: torch.Tensor,
    symmetric: bool = True,
    per_channel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight matrix (out, in) to INT8.

    Returns (w_int8, scales) where scales are FP16 and broadcast to (out,) when
    per_channel else a scalar tensor.
    """
    assert w.ndim == 2, "weight must be 2-D (out, in)"
    assert symmetric, "only symmetric quantization implemented"
    if per_channel:
        amax = w.abs().amax(dim=1).clamp_min(1e-8)
    else:
        amax = w.abs().amax().clamp_min(1e-8).expand(w.shape[0])
    scales = amax / 127.0
    w_int8 = (w / scales.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scales.to(torch.float16)
