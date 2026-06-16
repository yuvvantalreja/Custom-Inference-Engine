from __future__ import annotations

import argparse

import torch

from engine.device import require_cuda


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["flash", "int8", "spec", "smoke", "all"], default="all")
    args = ap.parse_args()

    require_cuda()

    run_flash = args.only in ("all", "flash")
    run_int8 = args.only in ("all", "int8")
    run_spec = args.only in ("all", "spec")
    run_smoke = args.only in ("all", "smoke")

    if run_flash:
        from benchmarks.bench_flash_attention import run as f
        f()
    if run_int8:
        from benchmarks.bench_int8_matmul import run as f
        f()
    if run_spec:
        from benchmarks.bench_speculative import run as f
        f()
    if run_smoke:
        from engine.config import EngineConfig, ModelConfig
        from engine.dtype import activation_dtype
        from engine.model.loader import randomize_weights
        from engine.runtime import build_engine, cast_non_quantized_params
        from engine.tokenizer import CharTokenizer

        tok = CharTokenizer()
        model_cfg = ModelConfig(
            vocab_size=tok.vocab_size, hidden_size=32, num_layers=2,
            num_heads=2, num_kv_heads=2, head_dim=16,
            max_position_embeddings=64, intermediate_size=64,
        )
        engine_cfg = EngineConfig(num_kv_blocks=16, kv_block_size=8)
        engine = build_engine(model_cfg, engine_cfg, tok)
        randomize_weights(engine.target, seed=0)
        cast_non_quantized_params(engine.target, activation_dtype(require_cuda()))

        s = engine.generate("hi", max_new_tokens=4).collect()
        print(f"smoke: {s!r}")
        print(f"smoke peak VRAM: {torch.cuda.max_memory_allocated()/2**20:.1f} MiB")


if __name__ == "__main__":
    main()
