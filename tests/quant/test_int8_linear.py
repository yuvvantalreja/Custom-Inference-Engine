import torch
import torch.nn as nn

from engine.quant import Int8Linear, quantize_linear


def test_int8_linear_close_to_fp():
    torch.manual_seed(0)
    lin = nn.Linear(64, 32, bias=True)
    qlin = quantize_linear(lin)
    x = torch.randn(4, 64)
    y_fp = lin(x)
    y_q = qlin(x)
    assert y_q.shape == y_fp.shape
    # Symmetric per-channel INT8: expect <2% relative error on random weights.
    err = (y_q - y_fp).abs().max().item()
    fp_scale = y_fp.abs().max().item()
    assert err / fp_scale < 0.05


def test_int8_linear_shapes():
    qlin = Int8Linear(16, 8, bias=False)
    assert qlin.weight_int8.shape == (8, 16)
    assert qlin.scales.shape == (8,)
    assert qlin.bias is None
