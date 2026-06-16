import os

from engine.config import EngineConfig
from engine.model.loader import load_hf
from engine.runtime import build_engine, cast_non_quantized_params
from engine.dtype import activation_dtype


def start_engine():
    model_cfg, target, tok = load_hf()

    blocks_needed = (512 // 16) + 8
    engine_cfg = EngineConfig(
        num_kv_blocks=max(blocks_needed, 64),
        kv_block_size=16,
        max_seq_len=512,
    )

    engine = build_engine(model_cfg, engine_cfg, tok, target=target)

    return engine


def conversation_loop():
    engine = start_engine()
    # engine._get_cache_stats()
    while True:
        user_input = input("Write some text...")
        # user_input += '\n\n'
        output = engine.generate(user_input, max_new_tokens=50).collect()
        print(f"model output: ", output)
        engine._get_cache_stats()


if __name__ == '__main__':
    conversation_loop()
            