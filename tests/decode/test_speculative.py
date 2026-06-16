from engine.decode.base_loop import generate
from engine.decode.speculative import speculative_generate
from engine.model import build_target
from engine.model.loader import randomize_weights
from engine.runtime import cast_non_quantized_params
from engine.dtype import activation_dtype
from tests.conftest import make_cache, tiny_cfg


def _clone_model(cfg, seed, device):
    """Build two structurally identical models with the same weights."""
    m1 = randomize_weights(build_target(cfg), seed=seed).to(device)
    cast_non_quantized_params(m1, activation_dtype(device))
    m2 = build_target(cfg).to(device)
    m2.load_state_dict(m1.state_dict())
    for mm1, mm2 in zip(m1.modules(), m2.modules()):
        from engine.quant import Int8Linear
        if isinstance(mm1, Int8Linear):
            mm2.weight_int8.copy_(mm1.weight_int8)
            mm2.scales.copy_(mm1.scales)
    cast_non_quantized_params(m2, activation_dtype(device))
    return m1, m2


def test_speculative_matches_base_loop_when_draft_equals_target(device):
    cfg = tiny_cfg()
    target, draft = _clone_model(cfg, seed=3, device=device)

    prompt = [1, 4, 2]
    base = list(generate(target, prompt, make_cache(cfg, device), max_new_tokens=8))

    target2, draft2 = _clone_model(cfg, seed=3, device=device)
    spec = list(speculative_generate(
        target2, draft2, prompt, make_cache(cfg, device), make_cache(cfg, device),
        max_new_tokens=8, gamma=3,
    ))
    assert spec == base
