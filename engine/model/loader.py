from __future__ import annotations

import torch
import torch.nn as nn

from engine.model.transformer import TransformerLM
from engine.quant import quantize_linear


def quantize_in_place(model: nn.Module) -> None:
    """Replace every ``nn.Linear`` in ``model`` with an ``Int8Linear``."""
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(model, name, quantize_linear(child))
        else:
            quantize_in_place(child)


def randomize_weights(model: TransformerLM, seed: int = 0) -> TransformerLM:
    """Initialize a tiny model with deterministic pseudo-random weights.

    Useful for tests — real model loading would come from a checkpoint loader
    that unpacks safetensors and calls ``quantize_in_place``.
    """
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        with torch.no_grad():
            p.copy_(torch.empty_like(p).uniform_(-0.02, 0.02, generator=g))
    # Int8Linear weights live in buffers: re-calibrate from a random FP init.
    for mod in model.modules():
        from engine.quant import Int8Linear
        if isinstance(mod, Int8Linear):
            w_fp = torch.empty(mod.out_features, mod.in_features).uniform_(-0.02, 0.02, generator=g)
            from engine.quant.calibrate import calibrate_weight
            w_int8, scales = calibrate_weight(w_fp)
            mod.weight_int8.copy_(w_int8)
            mod.scales.copy_(scales)
    return model
