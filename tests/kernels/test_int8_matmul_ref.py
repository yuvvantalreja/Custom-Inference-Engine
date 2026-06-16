import pytest
import torch

from kernels.ref import int8_matmul_ref, dequantize_int8

pytestmark = pytest.mark.no_cuda


def test_per_channel_roundtrip_close():
    torch.manual_seed(0)
    out_f, in_f = 16, 32
    w_fp = torch.randn(out_f, in_f)
    # symmetric per-channel quantize
    scales = w_fp.abs().amax(dim=1) / 127.0
    scales = scales.clamp_min(1e-8)
    w_int8 = (w_fp / scales.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    w_dq = dequantize_int8(w_int8, scales)
    assert (w_dq - w_fp).abs().max() < scales.max() * 1.1  # within 1 quant step


def test_matmul_matches_fp():
    torch.manual_seed(0)
    out_f, in_f = 8, 16
    w_fp = torch.randn(out_f, in_f)
    scales = w_fp.abs().amax(dim=1) / 127.0
    scales = scales.clamp_min(1e-8)
    w_int8 = (w_fp / scales.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)

    x = torch.randn(4, in_f)
    y_q = int8_matmul_ref(x, w_int8, scales)
    w_dq = dequantize_int8(w_int8, scales)
    y_ref = x @ w_dq.t()
    assert torch.allclose(y_q, y_ref, atol=1e-5)


def test_grouped_quantization():
    torch.manual_seed(0)
    out_f, in_f, gs = 4, 32, 8
    w_fp = torch.randn(out_f, in_f)
    groups = w_fp.view(out_f, in_f // gs, gs)
    scales = groups.abs().amax(dim=-1) / 127.0
    scales = scales.clamp_min(1e-8)
    w_int8 = (groups / scales.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8).view(out_f, in_f)
    w_dq = dequantize_int8(w_int8, scales, group_size=gs)
    assert (w_dq - w_fp).abs().max() < scales.max() * 1.1
