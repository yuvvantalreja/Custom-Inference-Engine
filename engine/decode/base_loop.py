from __future__ import annotations

from typing import Callable, Iterator

import torch

from engine.cache.kv_cache import PagedKVCache
from engine.decode.sampler import Sampler, greedy_sample
from engine.model.transformer import TransformerLM


def generate(
    model: TransformerLM,
    prompt: list[int],
    cache: PagedKVCache,
    max_new_tokens: int,
    sampler: Sampler | None = None,
    stop_tokens: set[int] | None = None,
) -> Iterator[int]:
    """Yield generated token ids (excluding the prompt)."""
    device = next(model.parameters()).device
    table = cache.new_sequence()
    tokens = torch.tensor([prompt], device=device, dtype=torch.long)
    with torch.no_grad():
        logits = model(tokens, cache, table)
    next_id = greedy_sample(logits[0, -1]) if sampler is None else sampler.sample(logits[0, -1])
    for _ in range(max_new_tokens):
        if stop_tokens is not None and next_id in stop_tokens:
            return
        yield next_id
        t = torch.tensor([[next_id]], device=device, dtype=torch.long)
        with torch.no_grad():
            logits = model(t, cache, table)
        next_id = greedy_sample(logits[0, -1]) if sampler is None else sampler.sample(logits[0, -1])
