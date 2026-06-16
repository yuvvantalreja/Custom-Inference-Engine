from __future__ import annotations

import torch
import torch.nn as nn

from engine.config import ModelConfig
from engine.device import require_cuda
from engine.dtype import activation_dtype
from engine.model.transformer import TransformerLM, build_target
from engine.quant import Int8Linear
from engine.quant.calibrate import calibrate_weight
from engine.runtime import cast_non_quantized_params
from engine.tokenizer import HFTokenizer, Tokenizer


def quantize_in_place(model: nn.Module) -> None:
    """Replace every nn.Linear in model with an Int8Linear."""
    from engine.quant import quantize_linear

    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(model, name, quantize_linear(child))
        else:
            quantize_in_place(child)


def hf_config_to_model_config(hf_cfg) -> ModelConfig:
    """Map a HuggingFace Llama config to ModelConfig."""
    num_kv_heads = getattr(hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads)
    head_dim = getattr(hf_cfg, "head_dim", None)
    if head_dim is None and hasattr(hf_cfg, "hidden_size"):
        head_dim = hf_cfg.hidden_size // hf_cfg.num_attention_heads
    return ModelConfig(
        vocab_size=hf_cfg.vocab_size,
        hidden_size=hf_cfg.hidden_size,
        num_layers=hf_cfg.num_hidden_layers,
        num_heads=hf_cfg.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=hf_cfg.intermediate_size,
        max_position_embeddings=hf_cfg.max_position_embeddings,
        rope_theta=getattr(hf_cfg, "rope_theta", 10000.0),
        rms_norm_eps=getattr(hf_cfg, "rms_norm_eps", 1e-5),
        tie_word_embeddings=getattr(hf_cfg, "tie_word_embeddings", False),
    )


def load_int8_linear(dst: Int8Linear, weight: torch.Tensor) -> None:
    """Quantize HF FP weight matrix into an Int8Linear module."""
    w_int8, scales = calibrate_weight(weight.detach().float().cpu())
    dst.weight_int8.copy_(w_int8.to(dst.weight_int8.device))
    dst.scales.copy_(scales.to(device=dst.scales.device, dtype=dst.scales.dtype))


def copy_hf_weights(hf_model, engine: TransformerLM) -> None:
    """Copy Llama weights from a HF LlamaForCausalLM into TransformerLM.
    """
    hf_inner = hf_model.model
    device = next(engine.parameters()).device

    engine.embed.weight.data.copy_(
        hf_inner.embed_tokens.weight.detach().to(device=device, dtype=engine.embed.weight.dtype)
    )

    for i, hf_layer in enumerate(hf_inner.layers):
        eng_layer = engine.layers[i]
        eng_layer.attn_norm.weight.data.copy_(
            # detach hf model weights from training backprop graph
            hf_layer.input_layernorm.weight.detach().to(
                device=device, dtype=eng_layer.attn_norm.weight.dtype
            )
        )
        eng_layer.mlp_norm.weight.data.copy_(
            hf_layer.post_attention_layernorm.weight.detach().to(
                device=device, dtype=eng_layer.mlp_norm.weight.dtype
            )
        )

        attn = hf_layer.self_attn
        load_int8_linear(eng_layer.attn.q_proj, attn.q_proj.weight)
        load_int8_linear(eng_layer.attn.k_proj, attn.k_proj.weight)
        load_int8_linear(eng_layer.attn.v_proj, attn.v_proj.weight)
        load_int8_linear(eng_layer.attn.o_proj, attn.o_proj.weight)

        mlp = hf_layer.mlp
        load_int8_linear(eng_layer.mlp.gate, mlp.gate_proj.weight)
        load_int8_linear(eng_layer.mlp.up, mlp.up_proj.weight)
        load_int8_linear(eng_layer.mlp.down, mlp.down_proj.weight)

    engine.final_norm.weight.data.copy_(
        hf_inner.norm.weight.detach().to(device=device, dtype=engine.final_norm.weight.dtype)
    )

    if engine.cfg.tie_word_embeddings:
        load_int8_linear(engine.lm_head, engine.embed.weight.detach())
    elif hasattr(hf_model, "lm_head") and hf_model.lm_head is not None:
        load_int8_linear(engine.lm_head, hf_model.lm_head.weight)
    else:
        load_int8_linear(engine.lm_head, engine.embed.weight.detach())


def load_hf(
    name: str = "meta-llama/Llama-3.2-1B",
    device: str | torch.device | None = None,
) -> tuple[ModelConfig, TransformerLM, Tokenizer]:
    """Load a Llama-family HF checkpoint into this engine's TransformerLM.

    Returns (model_cfg, loaded_model, tokenizer)
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # dev = require_cuda(device)
    dev = 'cpu'
    act_dtype = activation_dtype(dev)

    hf_cfg = AutoConfig.from_pretrained(name)

    # https://huggingface.co/docs/transformers/en/model_doc/llama
    model_cfg = hf_config_to_model_config(hf_cfg)

    if model_cfg.head_dim is not None and model_cfg.head_dim < 16:
        raise ValueError(
            f"head_dim={model_cfg.head_dim} is too small for the Triton attention "
            f"kernel (minimum 16); pick a different checkpoint"
        )

    engine = build_target(model_cfg).to(dev)
    cast_non_quantized_params(engine, act_dtype)

    hf_model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=act_dtype,
        low_cpu_mem_usage=True,
        cache_dir="weights"
    )

    print(f"{hf_model}")

    for name, _ in hf_model.named_modules(): # every submodule path
        print(f"{name}")

    copy_hf_weights(hf_model, engine)
    del hf_model
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    hf_tok = AutoTokenizer.from_pretrained(name)
    return model_cfg, engine, HFTokenizer(hf_tok)


def randomize_weights(model: TransformerLM, seed: int = 0) -> TransformerLM:
    """Initialize a tiny model with deterministic pseudo-random weights.

    Useful for tests — real model loading uses load_hf instead.
    """
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        with torch.no_grad():
            p.copy_(torch.empty_like(p).uniform_(-0.02, 0.02, generator=g))
    for mod in model.modules():
        if isinstance(mod, Int8Linear):
            w_fp = torch.empty(mod.out_features, mod.in_features).uniform_(-0.02, 0.02, generator=g)
            w_int8, scales = calibrate_weight(w_fp)
            mod.weight_int8.copy_(w_int8)
            mod.scales.copy_(scales)
    return model


if __name__ == '__main__':
    load_hf()