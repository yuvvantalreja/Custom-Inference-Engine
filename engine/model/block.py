from __future__ import annotations

import math

import torch
import torch.nn as nn

from engine.attention import flash_attention
from engine.cache.block_table import BlockTable
from engine.cache.kv_cache import PagedKVCache
from engine.model.mlp import SwiGLU
from engine.model.norm import RMSNorm
from engine.model.rope import RotaryEmbedding
from engine.quant import Int8Linear


class Attention(nn.Module):
    """Multi-head (optionally grouped-query) attention with paged KV cache."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope: RotaryEmbedding,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = Int8Linear(hidden_size, num_heads * head_dim)
        self.k_proj = Int8Linear(hidden_size, num_kv_heads * head_dim)
        self.v_proj = Int8Linear(hidden_size, num_kv_heads * head_dim)
        self.o_proj = Int8Linear(num_heads * head_dim, hidden_size)
        self.rope = rope

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        cache: PagedKVCache,
        table: BlockTable,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """x: (1, L, D). positions: (L,) absolute positions of each new token."""
        B, L, D = x.shape
        assert B == 1, "batch size must be 1 in v1"
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope.apply(q, positions=positions)
        k = self.rope.apply(k, positions=positions)

        # Append this step's k,v to the cache, then gather the full history.
        write_start = int(positions[0].item())
        cache.append(
            layer_idx,
            table,
            k[0].transpose(0, 1),
            v[0].transpose(0, 1),
            start=write_start,
        )
        full_k, full_v = cache.gather(layer_idx, table)  # (N, Hk, D)
        full_k = full_k.transpose(0, 1).unsqueeze(0)  # (1, Hk, N, D)
        full_v = full_v.transpose(0, 1).unsqueeze(0)

        out = flash_attention(
            q, full_k, full_v, causal=True, scale=1.0 / math.sqrt(self.head_dim)
        )
        out = (
            out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)
        )
        return self.o_proj(out)


class DecoderBlock(nn.Module):
    def __init__(self, cfg, rope: RotaryEmbedding):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attn = Attention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            rope=rope,
        )
        self.mlp_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x, layer_idx, cache, table, positions):
        x = x + self.attn(self.attn_norm(x), layer_idx, cache, table, positions)
        x = x + self.mlp(self.mlp_norm(x))
        return x
