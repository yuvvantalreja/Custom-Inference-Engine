import torch

from engine.model.rope import RotaryEmbedding


def test_rope_module_identity_at_zero():
    rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
    x = torch.randn(1, 2, 1, 16)
    y = rope.apply(x, positions=torch.tensor([0]))
    assert torch.allclose(y, x, atol=1e-6)


def test_rope_module_different_positions_differ():
    rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
    x = torch.randn(1, 2, 1, 16)
    y0 = rope.apply(x, positions=torch.tensor([0]))
    y5 = rope.apply(x, positions=torch.tensor([5]))
    assert not torch.allclose(y0, y5)
