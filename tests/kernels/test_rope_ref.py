import torch

from kernels.ref import build_rope_cache, apply_rotary_ref


def test_rope_zero_position_identity():
    cos, sin = build_rope_cache(8, 16)
    x = torch.randn(1, 2, 1, 16)
    # at position 0 rotary is identity (cos=1, sin=0)
    y = apply_rotary_ref(x, cos, sin, positions=torch.tensor([0]))
    assert torch.allclose(y, x, atol=1e-6)


def test_rope_preserves_shape_and_norm_pairwise():
    torch.manual_seed(0)
    cos, sin = build_rope_cache(32, 16)
    x = torch.randn(2, 4, 5, 16)
    y = apply_rotary_ref(x, cos, sin)
    assert y.shape == x.shape
    # rotary preserves paired-dimension L2 norm: x1^2+x2^2 == y1^2+y2^2
    D = 16
    x_pair = x[..., : D // 2] ** 2 + x[..., D // 2 :] ** 2
    y_pair = y[..., : D // 2] ** 2 + y[..., D // 2 :] ** 2
    assert torch.allclose(x_pair, y_pair, atol=1e-5)
