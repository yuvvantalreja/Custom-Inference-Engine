from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Sampler:
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None

    def __post_init__(self):
        self._generator: torch.Generator | None = None
        if self.seed is not None:
            self._generator = torch.Generator().manual_seed(self.seed)

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: (V,) -> probability distribution (V,)."""
        if self.temperature == 0.0:
            # Degenerate: place all mass on the argmax.
            p = torch.zeros_like(logits, dtype=torch.float32)
            p[int(torch.argmax(logits).item())] = 1.0
            return p

        x = logits.float() / self.temperature
        if self.top_k is not None and self.top_k > 0:
            kth = torch.topk(x, k=min(self.top_k, x.numel())).values.min()
            x = torch.where(x < kth, torch.full_like(x, float("-inf")), x)
        probs = torch.softmax(x, dim=-1)
        if self.top_p is not None and 0.0 < self.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep = cumulative - sorted_probs <= self.top_p
            filtered = torch.zeros_like(probs)
            filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
            probs = filtered / filtered.sum().clamp_min(1e-12)
        return probs

    def sample(self, logits: torch.Tensor) -> int:
        probs = self.probabilities(logits)
        if self.temperature == 0.0:
            return int(torch.argmax(probs).item())
        idx = torch.multinomial(probs, num_samples=1, generator=self._generator)
        return int(idx.item())


def greedy_sample(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits).item())
