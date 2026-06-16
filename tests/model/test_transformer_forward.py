import torch

from tests.conftest import make_cache, tiny_cfg, tiny_model


def test_forward_shape_and_cache_growth(device):
    cfg = tiny_cfg()
    model = tiny_model(cfg)
    cache = make_cache(cfg, device)
    table = cache.new_sequence()
    tokens = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    logits = model(tokens, cache, table)
    assert logits.shape == (1, 5, cfg.vocab_size)
    assert table.length == 5


def test_forward_is_deterministic(device):
    cfg = tiny_cfg()
    model = tiny_model(cfg, seed=42)
    tokens = torch.tensor([[7, 3, 1]], device=device)

    cache = make_cache(cfg, device)
    table = cache.new_sequence()
    a = model(tokens, cache, table)

    cache2 = make_cache(cfg, device)
    table2 = cache2.new_sequence()
    b = model(tokens, cache2, table2)
    assert torch.allclose(a, b)


def test_incremental_forward_matches_full(device):
    """Feed 3 tokens at once vs one-by-one; last-step logits must match."""
    cfg = tiny_cfg()
    model = tiny_model(cfg, seed=1)
    tokens = torch.tensor([[5, 2, 9]], device=device)

    cache_full = make_cache(cfg, device)
    table_full = cache_full.new_sequence()
    full = model(tokens, cache_full, table_full)

    cache_inc = make_cache(cfg, device)
    table_inc = cache_inc.new_sequence()
    for i in range(tokens.shape[1]):
        step_logits = model(tokens[:, i : i + 1], cache_inc, table_inc)
    assert torch.allclose(step_logits[0, -1], full[0, -1], atol=1e-3)
