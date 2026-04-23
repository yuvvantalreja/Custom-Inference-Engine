from __future__ import annotations

from typing import Callable, Iterable, Iterator

from engine.tokenizer import Tokenizer


class TokenStreamer:
    """Wrap a token-id iterable, yielding (token_id, new_text) pairs.

    Incremental detokenization is done by re-decoding the full running list
    and diffing against the previously emitted string — this is correct even
    for tokenizers where a single new id can change the decoding of earlier
    tokens (common with BPE/SentencePiece whitespace merges).
    """

    def __init__(self, tokenizer: Tokenizer, token_iter: Iterable[int]):
        self.tokenizer = tokenizer
        self._iter = iter(token_iter)
        self._ids: list[int] = []
        self._text: str = ""

    def __iter__(self) -> Iterator[tuple[int, str]]:
        return self

    def __next__(self) -> tuple[int, str]:
        tok = next(self._iter)
        self._ids.append(tok)
        full = self.tokenizer.decode(self._ids)
        new = full[len(self._text) :]
        self._text = full
        return tok, new

    def text(self) -> str:
        return self._text

    def collect(self, on_token: Callable[[int, str], None] | None = None) -> str:
        for tok, new in self:
            if on_token is not None:
                on_token(tok, new)
        return self._text
