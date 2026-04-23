from __future__ import annotations

import torch


def build_rope_cache(
    max_seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) of shape (max_seq_len, head_dim/2)."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    device = torch.device(device)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (L, D/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rotary_ref(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply rotary embeddings to x.

    x: (B, H, L, D) where D is even.
    cos, sin: (max_seq, D/2) cache.
    positions: (L,) or (B, L) integer positions to index into cos/sin.
               If None, positions default to [0, L).
    """
    B, H, L, D = x.shape
    if positions is None:
        positions = torch.arange(L, device=x.device)
    if positions.ndim == 1:
        c = cos[positions]  # (L, D/2)
        s = sin[positions]
        c = c.view(1, 1, L, D // 2)
        s = s.view(1, 1, L, D // 2)
    else:
        c = cos[positions]  # (B, L, D/2)
        s = sin[positions]
        c = c.view(B, 1, L, D // 2)
        s = s.view(B, 1, L, D // 2)

    x1 = x[..., : D // 2]
    x2 = x[..., D // 2 :]
    rot1 = x1 * c - x2 * s
    rot2 = x1 * s + x2 * c
    return torch.cat([rot1, rot2], dim=-1).to(x.dtype)
