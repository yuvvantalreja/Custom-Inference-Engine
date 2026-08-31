# Custom Inference Engine

GPU-first LLM inference engine focused on fast local generation with:

- Triton-backed attention and INT8 matmul kernels
- Paged KV cache for long-running generation loops
- INT8 linear quantization path for transformer layers
- Greedy speculative decoding (draft + target verification)
- Lightweight test/benchmark harnesses for kernels and runtime behavior

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA (the runtime is CUDA-only)
- PyTorch with CUDA support

## Installation

```bash
pip install -e ".[gpu,tokenizer,dev]"
```

Or install only base dependencies:

```bash
pip install -e .
```

## Quick Start

Run the interactive generation loop:

```bash
python /home/runner/work/Custom-Inference-Engine/Custom-Inference-Engine/api.py
```

This loads `meta-llama/Llama-3.2-1B` via Hugging Face, builds the engine, and starts a prompt loop.

## Programmatic Usage

Core flow:

1. Build or load a `ModelConfig`
2. Create an `EngineConfig`
3. Build the engine with `build_engine(...)`
4. Call `engine.generate(prompt, max_new_tokens=...)`

See:
- `/home/runner/work/Custom-Inference-Engine/Custom-Inference-Engine/api.py`
- `/home/runner/work/Custom-Inference-Engine/Custom-Inference-Engine/engine/runtime.py`
- `/home/runner/work/Custom-Inference-Engine/Custom-Inference-Engine/engine/model/loader.py`

## Testing

Run all tests:

```bash
pytest
```

Run only fast/no-CUDA import tests:

```bash
pytest -m no_cuda
```

Run integration test with a specific Hugging Face model:

```bash
HF_MODEL_ID=meta-llama/Llama-3.2-1B pytest -m integration
```

## Benchmarks

Run all benchmark suites:

```bash
python -m benchmarks.run --only all
```

Run individual suites:

```bash
python -m benchmarks.run --only flash
python -m benchmarks.run --only int8
python -m benchmarks.run --only spec
python -m benchmarks.run --only smoke
```

## Repository Layout

```text
engine/       Core runtime, model, decoding loop, quantization, tokenizer, cache
kernels/      Triton and reference kernels (flash attention, INT8 matmul, RoPE)
benchmarks/   Benchmark entrypoints for kernel/runtime performance checks
tests/        Unit + integration tests for runtime, decode, cache, kernels, model
api.py        Interactive local inference entrypoint
quant.py      Quantization playground script
```
