from __future__ import annotations

import torch


def dequantize_int8(
    w_int8: torch.Tensor,
    scales: torch.Tensor,
    group_size: int | None = None,
) -> torch.Tensor:
    """Dequantize INT8 weights to FP32.

    Layouts:
      per-channel: w_int8 (out, in), scales (out,) -> (out, in) FP32
      per-group:   w_int8 (out, in), scales (out, in/group_size) -> (out, in) FP32
    """
    if group_size is None or scales.ndim == 1:
        return w_int8.float() * scales.float().unsqueeze(-1)
    out, in_ = w_int8.shape
    assert in_ % group_size == 0
    num_groups = in_ // group_size
    assert scales.shape == (out, num_groups)
    w = w_int8.float().view(out, num_groups, group_size)
    s = scales.float().view(out, num_groups, 1)
    return (w * s).view(out, in_)


def int8_matmul_ref(
    x: torch.Tensor,
    w_int8: torch.Tensor,
    scales: torch.Tensor,
    group_size: int | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """y = x @ dequantize(w_int8, scales).T (+ bias).

    x: (..., in_features)
    w_int8: (out_features, in_features) int8
    scales: per-channel (out,) or per-group (out, in/group)
    Returns: (..., out_features) in x.dtype.
    """
    in_dtype = x.dtype
    w = dequantize_int8(w_int8, scales, group_size=group_size)
    y = torch.matmul(x.float(), w.t())
    if bias is not None:
        y = y + bias.float()
    return y.to(in_dtype)
