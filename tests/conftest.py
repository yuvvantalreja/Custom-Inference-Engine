from __future__ import annotations

import pytest
import torch

from engine.cache.kv_cache import KVCacheConfig, PagedKVCache
from engine.config import ModelConfig
from engine.dtype import activation_dtype, kv_cache_dtype
from engine.model import build_target
from engine.model.loader import randomize_weights
from engine.runtime import cast_non_quantized_params


def pytest_collection_modifyitems(config, items):
    """Skip CUDA-required tests unless a GPU is available."""
    if torch.cuda.is_available():
        return
    skip_cuda = pytest.mark.skip(reason="CUDA required")
    for item in items:
        if item.get_closest_marker("no_cuda"):
            continue
        item.add_marker(skip_cuda)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda")


def tiny_cfg(**overrides) -> ModelConfig:
    cfg = ModelConfig(
        vocab_size=32,
        hidden_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        intermediate_size=64,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_cache(cfg: ModelConfig, device: torch.device | None = None) -> PagedKVCache:
    dev = device or torch.device("cuda")
    return PagedKVCache(KVCacheConfig(
        num_layers=cfg.num_layers,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        num_blocks=32,
        block_size=8,
        dtype=kv_cache_dtype(dev),
        device=dev,
    ))


def tiny_model(cfg: ModelConfig, seed: int = 0, device: torch.device | None = None):
    dev = device or torch.device("cuda")
    model = randomize_weights(build_target(cfg), seed=seed).to(dev)
    cast_non_quantized_params(model, activation_dtype(dev))
    return model
