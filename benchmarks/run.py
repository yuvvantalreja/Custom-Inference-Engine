from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["flash", "int8", "spec", "smoke", "all"], default="all")
    ap.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    ap.add_argument("--require-gpu", action="store_true",
                    help="Fail loudly if CUDA is unavailable instead of falling back to CPU.")
    args = ap.parse_args()

    have_cuda = torch.cuda.is_available()
    if args.require_gpu and not have_cuda:
        raise SystemExit("--require-gpu set but torch.cuda.is_available() is False")
    if args.device == "cuda" and not have_cuda:
        raise SystemExit("--device cuda requested but CUDA is not available")

    run_flash = args.only in ("all", "flash")
    run_int8 = args.only in ("all", "int8")
    run_spec = args.only in ("all", "spec")
    run_smoke = args.only in ("all", "smoke")

    if run_flash:
        from benchmarks.bench_flash_attention import run as f
        f(device=args.device)
    if run_int8:
        from benchmarks.bench_int8_matmul import run as f
        f(device=args.device)
    if run_spec:
        from benchmarks.bench_speculative import run as f
        f(device=args.device)
    if run_smoke:
        from engine.config import EngineConfig, ModelConfig
        from engine.model.loader import randomize_weights
        from engine.runtime import build_engine
        from engine.tokenizer import CharTokenizer

        use_cuda = (args.device == "cuda") or (args.device == "auto" and have_cuda)
        device_str = "cuda" if use_cuda else "cpu"
        tok = CharTokenizer()
        model_cfg = ModelConfig(vocab_size=tok.vocab_size, hidden_size=16, num_layers=2,
                                num_heads=2, num_kv_heads=2, max_position_embeddings=64,
                                intermediate_size=32)
        # Randomize BEFORE build_engine's dtype/device cast so weights land in FP16 on CUDA.
        engine_cfg = EngineConfig(num_kv_blocks=16, kv_block_size=8, device=device_str)
        engine = build_engine(model_cfg, engine_cfg, tok)
        randomize_weights(engine.target, seed=0)
        # Re-run the cast now that randomize_weights wrote FP32 weights.
        from engine.runtime import _cast_non_quantized_params
        from engine.dtype import activation_dtype, as_device
        _cast_non_quantized_params(engine.target, activation_dtype(as_device(device_str)))

        s = engine.generate("hi", max_new_tokens=4).collect()
        print(f"smoke[{device_str}]: {s!r}")
        if use_cuda:
            print(f"smoke peak VRAM: {torch.cuda.max_memory_allocated()/2**20:.1f} MiB")


if __name__ == "__main__":
    main()
