import torch

from engine.decode.sampler import Sampler, greedy_sample


def test_greedy_sample():
    logits = torch.tensor([0.1, 3.0, -1.0, 2.5])
    assert greedy_sample(logits) == 1


def test_temperature_zero_argmax():
    sampler = Sampler(temperature=0.0)
    logits = torch.tensor([0.1, 3.0, -1.0, 2.5])
    assert sampler.sample(logits) == 1


def test_seeded_sampler_is_reproducible():
    logits = torch.randn(32)
    s1 = Sampler(temperature=1.0, seed=123).sample(logits)
    s2 = Sampler(temperature=1.0, seed=123).sample(logits)
    assert s1 == s2


def test_top_k_masks_out_low_probability():
    logits = torch.tensor([0.0, 0.0, 100.0, 0.0, 0.0])
    sampler = Sampler(temperature=1.0, top_k=1, seed=0)
    for _ in range(5):
        assert sampler.sample(logits) == 2
