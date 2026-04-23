import torch

from engine.cache.kv_cache import PagedKVCache, KVCacheConfig
from engine.config import ModelConfig
from engine.decode.base_loop import generate
from engine.decode.speculative import speculative_generate
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


def _clone_model(cfg, seed):
    """Build two structurally identical models with the same weights."""
    m1 = randomize_weights(build_target(cfg), seed=seed)
    m2 = build_target(cfg)
    m2.load_state_dict(m1.state_dict())
    for mm1, mm2 in zip(m1.modules(), m2.modules()):
        from engine.quant import Int8Linear
        if isinstance(mm1, Int8Linear):
            mm2.weight_int8.copy_(mm1.weight_int8)
            mm2.scales.copy_(mm1.scales)
    return m1, m2


def test_speculative_matches_base_loop_when_draft_equals_target():
    cfg = tiny_cfg()
    target, draft = _clone_model(cfg, seed=3)

    prompt = [1, 4, 2]
    base = list(generate(target, prompt, make_cache(cfg), max_new_tokens=8))

    target2, draft2 = _clone_model(cfg, seed=3)
    spec = list(speculative_generate(
        target2, draft2, prompt, make_cache(cfg), make_cache(cfg),
        max_new_tokens=8, gamma=3,
    ))
    assert spec == base
