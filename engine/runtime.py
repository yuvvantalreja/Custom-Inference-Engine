from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from engine.cache.kv_cache import KVCacheConfig, PagedKVCache
from engine.config import EngineConfig, ModelConfig, SpecConfig
from engine.decode.base_loop import generate
from engine.decode.sampler import Sampler
from engine.decode.speculative import speculative_generate
from engine.dtype import activation_dtype, as_device, kv_cache_dtype
from engine.model import TransformerLM, build_draft, build_target
from engine.quant import Int8Linear
from engine.stream import TokenStreamer
from engine.tokenizer import Tokenizer


def _cast_non_quantized_params(model: nn.Module, dtype: torch.dtype) -> None:
    """Cast everything except Int8Linear INT8 weights to ``dtype``.

    INT8 weights stay INT8; their FP16 scales/bias are re-cast too.
    """
    for mod in model.modules():
        if isinstance(mod, Int8Linear):
            mod.scales.data = mod.scales.data.to(dtype)
            if mod.bias is not None:
                mod.bias.data = mod.bias.data.to(dtype)
            continue
        for name, p in mod.named_parameters(recurse=False):
            p.data = p.data.to(dtype)
        for name, b in mod.named_buffers(recurse=False):
            if b.is_floating_point():
                setattr(mod, name, b.to(dtype))


@dataclass
class InferenceEngine:
    target: TransformerLM
    tokenizer: Tokenizer
    engine_cfg: EngineConfig
    draft: Optional[TransformerLM] = None
    spec_cfg: Optional[SpecConfig] = None
    _target_cache: Optional[PagedKVCache] = field(default=None, init=False, repr=False)
    _draft_cache: Optional[PagedKVCache] = field(default=None, init=False, repr=False)

    def _make_cache(self, model: TransformerLM) -> PagedKVCache:
        cfg = model.cfg
        device = as_device(self.engine_cfg.device)
        return PagedKVCache(KVCacheConfig(
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            num_blocks=self.engine_cfg.num_kv_blocks,
            block_size=self.engine_cfg.kv_block_size,
            dtype=kv_cache_dtype(device),
            device=device,
        ))

    def _ensure_caches(self) -> None:
        if self._target_cache is None:
            self._target_cache = self._make_cache(self.target)
        if self.draft is not None and self._draft_cache is None:
            self._draft_cache = self._make_cache(self.draft)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        sampler: Sampler | None = None,
    ) -> TokenStreamer:
        self._ensure_caches()
        prompt_ids = self.tokenizer.encode(prompt)
        stop = {self.tokenizer.eos_id}

        # Reuse long-lived caches across generate() calls; clear any prior
        # sequence so the next prompt starts at position 0.
        for seq_attr in ("_last_target_seq", "_last_draft_seq"):
            prev = getattr(self, seq_attr, None)
            cache = self._target_cache if seq_attr == "_last_target_seq" else self._draft_cache
            if prev is not None and cache is not None:
                cache.free_sequence(prev)
                setattr(self, seq_attr, None)

        if self.draft is not None and self.spec_cfg is not None and self.spec_cfg.enabled:
            token_iter = speculative_generate(
                self.target, self.draft, prompt_ids,
                self._target_cache, self._draft_cache,
                max_new_tokens=max_new_tokens, gamma=self.spec_cfg.gamma,
                sampler=sampler, stop_tokens=stop,
            )
        else:
            token_iter = generate(
                self.target, prompt_ids, self._target_cache,
                max_new_tokens=max_new_tokens, sampler=sampler, stop_tokens=stop,
            )
        return TokenStreamer(self.tokenizer, token_iter)


def build_engine(
    model_cfg: ModelConfig,
    engine_cfg: EngineConfig,
    tokenizer: Tokenizer,
    draft_cfg: ModelConfig | None = None,
    spec_cfg: SpecConfig | None = None,
) -> InferenceEngine:
    device = as_device(engine_cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("engine_cfg.device='cuda' but CUDA is not available")

    target = build_target(model_cfg).to(device)
    draft = build_draft(model_cfg, draft_cfg).to(device) if draft_cfg is not None else None

    act_dtype = activation_dtype(device)
    _cast_non_quantized_params(target, act_dtype)
    if draft is not None:
        _cast_non_quantized_params(draft, act_dtype)

    return InferenceEngine(
        target=target, tokenizer=tokenizer, engine_cfg=engine_cfg,
        draft=draft, spec_cfg=spec_cfg,
    )
