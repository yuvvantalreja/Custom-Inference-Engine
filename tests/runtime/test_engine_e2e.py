import torch

from engine.config import EngineConfig, ModelConfig
from engine.model.loader import randomize_weights
from engine.runtime import build_engine
from engine.tokenizer import CharTokenizer


def test_engine_end_to_end_generates_text():
    tok = CharTokenizer()
    model_cfg = ModelConfig(
        vocab_size=tok.vocab_size, hidden_size=16, num_layers=2,
        num_heads=2, num_kv_heads=2, max_position_embeddings=64,
        intermediate_size=32,
    )
    engine_cfg = EngineConfig(num_kv_blocks=16, kv_block_size=8, max_seq_len=64)

    engine = build_engine(model_cfg, engine_cfg, tok)
    randomize_weights(engine.target, seed=7)

    streamer = engine.generate("hello", max_new_tokens=4)
    out = streamer.collect()
    assert len(out) > 0
    assert len(out) <= 4  # one char per token for CharTokenizer
