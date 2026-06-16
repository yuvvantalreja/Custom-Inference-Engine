from __future__ import annotations

import torch
import torch.nn as nn

from engine.quant.calibrate import calibrate_weight


class Int8Linear(nn.Module):
    """Linear layer with INT8-quantized weights and optional FP16 bias.

    Weights are stored as int8 (out_features, in_features); scales are FP16
    (out_features,). Activations pass through in their native dtype.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scales", torch.zeros(out_features, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, linear: nn.Linear) -> "Int8Linear":
        mod = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        w_int8, scales = calibrate_weight(linear.weight.detach())
        mod.weight_int8.copy_(w_int8)
        mod.scales.copy_(scales)
        if linear.bias is not None:
            mod.bias.data.copy_(linear.bias.detach().to(torch.float16))
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("Int8Linear forward requires CUDA tensors")
        from kernels.int8_matmul import int8_matmul
        return int8_matmul(x, self.weight_int8, self.scales, bias=self.bias)


def quantize_linear(linear: nn.Linear) -> Int8Linear:
    return Int8Linear.from_float(linear)
