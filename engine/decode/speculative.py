"""Speculative decoding (Leviathan et al. 2023).

The draft proposes gamma tokens autoregressively. The target verifies all
gamma positions in a single forward pass, then accepts the longest prefix
per the Leviathan rejection rule. On rejection, the target's KV cache is
rolled back to the last accepted position so future steps stay consistent.

In greedy mode (temperature == 0), acceptance reduces to: keep tokens while
argmax(target_logits) == draft_token. This gives exactness vs. the base
loop when draft == target.
"""

from __future__ import annotations

from typing import Iterator

import torch

from engine.cache.kv_cache import PagedKVCache
from engine.decode.sampler import Sampler, greedy_sample
from engine.model.transformer import TransformerLM


def _draft_propose(
    draft: TransformerLM,
    draft_cache: PagedKVCache,
    draft_table,
    last_token: int,
    gamma: int,
    device,
) -> tuple[list[int], list[torch.Tensor]]:
    """Advance the draft by gamma tokens greedily. Returns tokens + logits."""
    proposed: list[int] = []
    logits_list: list[torch.Tensor] = []
    t = torch.tensor([[last_token]], device=device, dtype=torch.long)
    for _ in range(gamma):
        with torch.no_grad():
            logits = draft(t, draft_cache, draft_table)
        step = logits[0, -1]
        logits_list.append(step)
        next_id = greedy_sample(step)
        proposed.append(next_id)
        t = torch.tensor([[next_id]], device=device, dtype=torch.long)
    return proposed, logits_list


def speculative_generate(
    target: TransformerLM,
    draft: TransformerLM,
    prompt: list[int],
    target_cache: PagedKVCache,
    draft_cache: PagedKVCache,
    max_new_tokens: int,
    gamma: int = 4,
    sampler: Sampler | None = None,
    stop_tokens: set[int] | None = None,
) -> Iterator[int]:
    """Greedy speculative decoding. sampler is reserved for future temperature
    support but this implementation is greedy-only — with draft == target it
    produces exactly the same sequence as base_loop.generate."""

    assert sampler is None or sampler.temperature == 0.0, "only greedy speculative supported in v1"
    device = next(target.parameters()).device

    target_table = target_cache.new_sequence()
    draft_table = draft_cache.new_sequence()

    # Prefill both models on the prompt.
    prompt_t = torch.tensor([prompt], device=device, dtype=torch.long)
    with torch.no_grad():
        tgt_logits = target(prompt_t, target_cache, target_table)
        draft(prompt_t, draft_cache, draft_table)
    last_token = greedy_sample(tgt_logits[0, -1])

    # The first post-prompt token is already decided by the target's prefill;
    # emit it so our output matches the base loop exactly.
    yield last_token
    produced = 1
    while produced < max_new_tokens:
        if stop_tokens is not None and last_token in stop_tokens:
            yield last_token
            return

        # 1) Draft proposes gamma tokens conditioned on last_token.
        proposed, _ = _draft_propose(draft, draft_cache, draft_table, last_token, gamma, device)

        # 2) Target verifies in one forward: input is [last_token, p0, p1, ..., p_{gamma-1}].
        verify_in = torch.tensor([[last_token] + proposed], device=device, dtype=torch.long)
        with torch.no_grad():
            verify_logits = target(verify_in, target_cache, target_table)
        # verify_logits[0, i] predicts the token AFTER position i. So
        #   verify_logits[0, 0] -> prediction given last_token  (should match proposed[0] to accept)
        #   verify_logits[0, j] -> prediction given proposed[j-1]
        # Under greedy: accept proposed[j] iff argmax(verify_logits[0, j]) == proposed[j].
        accepted = 0
        fallback_token = greedy_sample(verify_logits[0, 0])
        for j in range(gamma):
            tgt_pred = greedy_sample(verify_logits[0, j])
            if tgt_pred == proposed[j]:
                accepted += 1
                fallback_token = greedy_sample(verify_logits[0, j + 1]) if j + 1 < gamma + 1 else None
            else:
                fallback_token = tgt_pred
                break

        # 3) Emit accepted tokens + one "bonus" token (the target's correction or
        # the extra greedy token after a full-accept sequence).
        emitted_this_round = list(proposed[:accepted])
        if fallback_token is not None:
            emitted_this_round.append(fallback_token)

        # 4) Roll back the target KV cache to just after the accepted prefix.
        # After target forward, target_table.length advanced by (1 + gamma).
        # Positions written: prompt_len + [0..gamma]. We keep prompt + (1 + accepted).
        keep_total = target_table.length - (gamma - accepted)
        if keep_total < target_table.length:
            target_cache.rollback(target_table, keep_total)

        # 5) Roll back draft similarly: draft wrote (1 prompt last token already cached) +
        # gamma draft tokens during propose. Keep prefix of length prompt + 1 + accepted.
        keep_draft = draft_table.length - (gamma - accepted)
        if keep_draft < draft_table.length:
            draft_cache.rollback(draft_table, keep_draft)

        # 6) Yield, updating the "last_token" anchor for the next round.
        for tok in emitted_this_round:
            if produced >= max_new_tokens:
                return
            if stop_tokens is not None and tok in stop_tokens:
                return
            yield tok
            produced += 1
            last_token = tok
