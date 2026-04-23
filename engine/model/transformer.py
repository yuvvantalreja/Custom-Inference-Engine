from __future__ import annotations

import torch
import torch.nn as nn

from engine.cache.block_table import BlockTable
from engine.cache.kv_cache import PagedKVCache
from engine.config import ModelConfig
from engine.model.block import DecoderBlock
from engine.model.norm import RMSNorm
from engine.model.rope import RotaryEmbedding
from engine.quant import Int8Linear


class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_position_embeddings, theta=cfg.rope_theta)
        self.layers = nn.ModuleList([DecoderBlock(cfg, self.rope) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = Int8Linear(cfg.hidden_size, cfg.vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        cache: PagedKVCache,
        table: BlockTable,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """tokens: (1, L). Returns logits (1, L, vocab)."""
        B, L = tokens.shape
        assert B == 1
        if positions is None:
            start = table.length
            positions = torch.arange(start, start + L, device=tokens.device)
        x = self.embed(tokens)
        for i, layer in enumerate(self.layers):
            x = layer(x, i, cache, table, positions)
        x = self.final_norm(x)
        return self.lm_head(x)


def build_target(cfg: ModelConfig) -> TransformerLM:
    return TransformerLM(cfg)


def build_draft(target_cfg: ModelConfig, draft_cfg: ModelConfig) -> TransformerLM:
    """Construct a draft model. Must share the architecture family of target.

    Enforces: same vocab, same head_dim, same num_kv_heads ratio so cache
    geometry can be shared/compared. Layers, hidden_size, intermediate_size
    are expected to be smaller.
    """
    assert draft_cfg.vocab_size == target_cfg.vocab_size
    assert draft_cfg.head_dim == target_cfg.head_dim
    assert draft_cfg.num_layers <= target_cfg.num_layers
    assert draft_cfg.hidden_size <= target_cfg.hidden_size
    return TransformerLM(draft_cfg)
