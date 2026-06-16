import pytest
import torch

from engine.model.norm import RMSNorm

pytestmark = pytest.mark.no_cuda


def test_rmsnorm_unit_input():
    norm = RMSNorm(8)
    x = torch.ones(2, 3, 8)
    y = norm(x)
    assert torch.allclose(y, torch.ones_like(y), atol=1e-5)


def test_rmsnorm_matches_formula():
    torch.manual_seed(0)
    norm = RMSNorm(16, eps=1e-5)
    norm.weight.data.copy_(torch.arange(16).float() + 1.0)
    x = torch.randn(4, 16)
    y = norm(x)
    rms = (x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5).rsqrt()
    expected = (x.float() * rms * norm.weight.float()).to(x.dtype)
    assert torch.allclose(y, expected, atol=1e-5)
