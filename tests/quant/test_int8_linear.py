import pytest
import torch
import torch.nn as nn

from engine.dtype import activation_dtype
from engine.quant import Int8Linear, quantize_linear


def test_int8_linear_close_to_fp(device):
    torch.manual_seed(0)
    dtype = activation_dtype(device)
    lin = nn.Linear(64, 32, bias=True).to(device=device, dtype=dtype)
    qlin = quantize_linear(lin).to(device)
    x = torch.randn(4, 64, device=device, dtype=dtype)
    y_fp = lin(x)
    y_q = qlin(x)
    assert y_q.shape == y_fp.shape
    err = (y_q - y_fp).abs().max().item()
    fp_scale = y_fp.abs().max().item()
    assert err / fp_scale < 0.05


@pytest.mark.no_cuda
def test_int8_linear_shapes():
    qlin = Int8Linear(16, 8, bias=False)
    assert qlin.weight_int8.shape == (8, 16)
    assert qlin.scales.shape == (8,)
    assert qlin.bias is None
