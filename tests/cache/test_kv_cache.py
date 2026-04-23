import torch

from engine.cache.kv_cache import PagedKVCache, KVCacheConfig


def make_cache(num_blocks=8, block_size=4, layers=2, heads=2, dim=4):
    return PagedKVCache(KVCacheConfig(
        num_layers=layers, num_kv_heads=heads, head_dim=dim,
        num_blocks=num_blocks, block_size=block_size,
    ))


def test_append_and_gather_roundtrip():
    torch.manual_seed(0)
    cache = make_cache()
    seq = cache.new_sequence()
    k = torch.randn(6, 2, 4)
    v = torch.randn(6, 2, 4)
    cache.append(0, seq, k, v)
    gk, gv = cache.gather(0, seq)
    assert torch.allclose(gk, k)
    assert torch.allclose(gv, v)
    assert seq.length == 6


def test_append_spans_multiple_blocks():
    cache = make_cache(block_size=2)
    seq = cache.new_sequence()
    k = torch.randn(5, 2, 4)
    v = torch.randn(5, 2, 4)
    cache.append(0, seq, k, v)
    assert len(seq.blocks) == 3  # ceil(5/2)
    gk, gv = cache.gather(0, seq)
    assert torch.allclose(gk, k)
    assert torch.allclose(gv, v)


def test_free_sequence_returns_blocks():
    cache = make_cache(num_blocks=4, block_size=2)
    seq = cache.new_sequence()
    cache.append(0, seq, torch.randn(4, 2, 4), torch.randn(4, 2, 4))
    assert len(cache._free) == 2
    cache.free_sequence(seq)
    assert len(cache._free) == 4
    assert seq.length == 0


def test_fork_is_deep_copy():
    cache = make_cache()
    seq = cache.new_sequence()
    cache.append(0, seq, torch.randn(3, 2, 4), torch.randn(3, 2, 4))
    forked = cache.fork(seq)
    gk0, _ = cache.gather(0, seq)
    gkf, _ = cache.gather(0, forked)
    assert torch.allclose(gk0, gkf)
    # mutate parent: fork unaffected
    cache.append(0, seq, torch.randn(1, 2, 4), torch.randn(1, 2, 4))
    assert seq.length == 4
    assert forked.length == 3
    gkf2, _ = cache.gather(0, forked)
    assert torch.allclose(gkf, gkf2)


def test_rollback_frees_blocks():
    cache = make_cache(num_blocks=4, block_size=2)
    seq = cache.new_sequence()
    cache.append(0, seq, torch.randn(4, 2, 4), torch.randn(4, 2, 4))
    assert len(cache._free) == 2
    cache.rollback(seq, new_length=2)
    assert seq.length == 2
    assert len(cache._free) == 3
