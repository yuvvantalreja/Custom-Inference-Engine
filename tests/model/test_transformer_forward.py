import torch

from engine.cache.kv_cache import PagedKVCache, KVCacheConfig
from engine.config import ModelConfig
from engine.model import build_target
from engine.model.loader import randomize_weights


def tiny_cfg():
    return ModelConfig(
        vocab_size=32, hidden_size=16, num_layers=2, num_heads=2,
        num_kv_heads=2, max_position_embeddings=32, intermediate_size=32,
    )


def make_cache(cfg):
    return PagedKVCache(KVCacheConfig(
        num_layers=cfg.num_layers, num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim,
        num_blocks=8, block_size=4,
    ))


def test_forward_shape_and_cache_growth():
    cfg = tiny_cfg()
    model = randomize_weights(build_target(cfg))
    cache = make_cache(cfg)
    table = cache.new_sequence()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    logits = model(tokens, cache, table)
    assert logits.shape == (1, 5, cfg.vocab_size)
    assert table.length == 5


def test_forward_is_deterministic():
    cfg = tiny_cfg()
    model = randomize_weights(build_target(cfg), seed=42)
    tokens = torch.tensor([[7, 3, 1]])

    cache = make_cache(cfg)
    table = cache.new_sequence()
    a = model(tokens, cache, table)

    cache2 = make_cache(cfg)
    table2 = cache2.new_sequence()
    b = model(tokens, cache2, table2)
    assert torch.allclose(a, b)


def test_incremental_forward_matches_full():
    """Feed 3 tokens at once vs one-by-one; last-step logits must match."""
    cfg = tiny_cfg()
    model = randomize_weights(build_target(cfg), seed=1)
    tokens = torch.tensor([[5, 2, 9]])

    cache_full = make_cache(cfg)
    table_full = cache_full.new_sequence()
    full = model(tokens, cache_full, table_full)

    cache_inc = make_cache(cfg)
    table_inc = cache_inc.new_sequence()
    for i in range(tokens.shape[1]):
        step_logits = model(tokens[:, i : i + 1], cache_inc, table_inc)
    assert torch.allclose(step_logits[0, -1], full[0, -1], atol=1e-4)
