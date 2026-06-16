import os

import pytest

from engine.config import EngineConfig
from engine.model.loader import load_hf, randomize_weights
from engine.runtime import build_engine, cast_non_quantized_params
from engine.dtype import activation_dtype
from engine.tokenizer import CharTokenizer
# from tests.conftest import tiny_cfg

# Override with e.g. HF_MODEL_ID=meta-llama/Llama-3.2-1B
HF_MODEL_ID = os.environ.get(
    "HF_MODEL_ID", "meta-llama/Llama-3.2-1B"
)


def test_engine_end_to_end_generates_text(device):
    tok = CharTokenizer()
    model_cfg = tiny_cfg(vocab_size=tok.vocab_size)
    engine_cfg = EngineConfig(num_kv_blocks=16, kv_block_size=8, max_seq_len=64)

    engine = build_engine(model_cfg, engine_cfg, tok)
    randomize_weights(engine.target, seed=7)
    cast_non_quantized_params(engine.target, activation_dtype(device))

    streamer = engine.generate("hello", max_new_tokens=4)
    out = streamer.collect()
    assert len(out) > 0
    assert len(out) <= 4


@pytest.mark.integration
def test_hf_model(device):
    """End-to-end generation with real HF Llama weights in TransformerLM."""
    # pytest.importorskip("transformers")

    model_cfg, target, tok = load_hf(HF_MODEL_ID, device=device)
    blocks_needed = (512 // 16) + 8
    engine_cfg = EngineConfig(
        num_kv_blocks=max(blocks_needed, 64),
        kv_block_size=16,
        max_seq_len=512,
    )

    engine = build_engine(model_cfg, engine_cfg, tok, target=target)
    out = engine.generate("Steven Paul Jobs was an American businessman, inventor, and investor. A pioneer of the ", max_new_tokens=50).collect()
    print(f"{out}")
    assert isinstance(out, str)
    assert len(out) > 0

if __name__ == "__main__":
    test_hf_model("cuda:0")