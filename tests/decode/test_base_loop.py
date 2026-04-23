import torch

from engine.cache.kv_cache import PagedKVCache, KVCacheConfig
from engine.config import ModelConfig
from engine.decode.base_loop import generate
from engine.model import build_target
from engine.model.loader import randomize_weights


def tiny_cfg():
    return ModelConfig(
        vocab_size=32, hidden_size=16, num_layers=2, num_heads=2,
        num_kv_heads=2, max_position_embeddings=64, intermediate_size=32,
    )


def make_cache(cfg):
    return PagedKVCache(KVCacheConfig(
        num_layers=cfg.num_layers, num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim, num_blocks=32, block_size=8,
    ))


def test_generate_is_deterministic_greedy():
    cfg = tiny_cfg()
    model = randomize_weights(build_target(cfg), seed=0)
    prompt = [1, 2, 3]

    out1 = list(generate(model, prompt, make_cache(cfg), max_new_tokens=5))
    out2 = list(generate(model, prompt, make_cache(cfg), max_new_tokens=5))
    assert out1 == out2
    assert len(out1) == 5


def test_generate_stops_at_stop_token():
    cfg = tiny_cfg()
    model = randomize_weights(build_target(cfg), seed=1)
    gen = generate(model, [1, 2], make_cache(cfg), max_new_tokens=5)
    first = next(gen)
    # Force stop on whatever first token is.
    gen2 = generate(model, [1, 2], make_cache(cfg), max_new_tokens=5, stop_tokens={first})
    out = list(gen2)
    assert out == []
