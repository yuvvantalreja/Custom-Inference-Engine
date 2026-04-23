from __future__ import annotations

import time

import torch

from engine.cache.kv_cache import KVCacheConfig, PagedKVCache
from engine.config import ModelConfig
from engine.decode.base_loop import generate
from engine.decode.speculative import speculative_generate
from engine.dtype import kv_cache_dtype
from engine.model import build_target
from engine.model.loader import randomize_weights
from engine.runtime import _cast_non_quantized_params


def run(max_new_tokens: int = 64, device: str = "auto") -> None:
    use_cuda = (device == "cuda") or (device == "auto" and torch.cuda.is_available())
    dev = torch.device("cuda" if use_cuda else "cpu")

    cfg = ModelConfig(vocab_size=128, hidden_size=32, num_layers=4, num_heads=4,
                      num_kv_heads=4, max_position_embeddings=128, intermediate_size=64)

    def fresh_cache():
        return PagedKVCache(KVCacheConfig(
            num_layers=cfg.num_layers, num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim, num_blocks=32, block_size=8,
            dtype=kv_cache_dtype(dev), device=dev,
        ))

    target = randomize_weights(build_target(cfg), seed=0).to(dev)
    draft = build_target(cfg).to(dev)
    draft.load_state_dict(target.state_dict())
    if use_cuda:
        _cast_non_quantized_params(target, torch.float16)
        _cast_non_quantized_params(draft, torch.float16)

    prompt = [1, 2, 3, 4]

    # Warmup (triggers Triton compile on CUDA).
    list(generate(target, prompt, fresh_cache(), max_new_tokens=4))
    if use_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    base_out = list(generate(target, prompt, fresh_cache(), max_new_tokens=max_new_tokens))
    if use_cuda:
        torch.cuda.synchronize()
    dt_base = time.perf_counter() - t0

    t0 = time.perf_counter()
    spec_out = list(speculative_generate(target, draft, prompt, fresh_cache(), fresh_cache(),
                                          max_new_tokens=max_new_tokens, gamma=4))
    if use_cuda:
        torch.cuda.synchronize()
    dt_spec = time.perf_counter() - t0

    print(f"base loop:   {dt_base*1000:.1f} ms, {len(base_out)} toks, {len(base_out)/dt_base:.1f} tok/s")
    print(f"speculative: {dt_spec*1000:.1f} ms, {len(spec_out)} toks, {len(spec_out)/dt_spec:.1f} tok/s")
    print(f"match: {base_out == spec_out}")
