"""Minimal tokenizer abstraction.

Real deployments wrap HuggingFace tokenizers or sentencepiece. For
testability without external deps we provide CharTokenizer which maps
single characters to ids — good enough for round-trip tests.
"""

from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    vocab_size: int
    eos_id: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def decode_incremental(self, ids: list[int], prev_text: str) -> str:
        """Return the *newly appended* text given a running decoded prefix."""
        ...


class CharTokenizer:
    """Single-char tokenizer over a fixed alphabet + PAD + EOS."""

    def __init__(self, alphabet: str | None = None):
        if alphabet is None:
            alphabet = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?'\n"
        self._chars = list(alphabet)
        self.pad_id = 0
        self.eos_id = 1
        self._char_to_id = {c: i + 2 for i, c in enumerate(self._chars)}
        self._id_to_char = {i + 2: c for i, c in enumerate(self._chars)}
        self.vocab_size = len(self._chars) + 2

    def encode(self, text: str) -> list[int]:
        return [self._char_to_id[c] for c in text if c in self._char_to_id]

    def decode(self, ids: list[int]) -> str:
        return "".join(
            self._id_to_char.get(i, "")
            for i in ids
            if i != self.pad_id and i != self.eos_id
        )

    def decode_incremental(self, ids: list[int], prev_text: str) -> str:
        full = self.decode(ids)
        return full[len(prev_text) :]


class HFTokenizer:
    """Thin wrapper around a HuggingFace PreTrainedTokenizer."""

    def __init__(self, tok):
        self._tok = tok
        self.vocab_size = int(getattr(tok, "vocab_size", len(tok.get_vocab())))
        eos = tok.eos_token_id
        if eos is None:
            raise ValueError("HF tokenizer must define eos_token_id")
        self.eos_id = int(eos)

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    def decode_incremental(self, ids: list[int], prev_text: str) -> str:
        full = self.decode(ids)
        return full[len(prev_text) :]
