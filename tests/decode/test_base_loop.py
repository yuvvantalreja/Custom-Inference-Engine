from engine.decode.base_loop import generate
from tests.conftest import make_cache, tiny_cfg, tiny_model


def test_generate_is_deterministic_greedy(device):
    cfg = tiny_cfg()
    model = tiny_model(cfg, seed=0)
    prompt = [1, 2, 3]

    out1 = list(generate(model, prompt, make_cache(cfg, device), max_new_tokens=5))
    out2 = list(generate(model, prompt, make_cache(cfg, device), max_new_tokens=5))
    assert out1 == out2
    assert len(out1) == 5


def test_generate_stops_at_stop_token(device):
    cfg = tiny_cfg()
    model = tiny_model(cfg, seed=1)
    gen = generate(model, [1, 2], make_cache(cfg, device), max_new_tokens=5)
    first = next(gen)
    gen2 = generate(model, [1, 2], make_cache(cfg, device), max_new_tokens=5, stop_tokens={first})
    out = list(gen2)
    assert out == []
