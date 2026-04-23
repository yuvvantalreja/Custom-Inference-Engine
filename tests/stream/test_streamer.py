from engine.stream import TokenStreamer
from engine.tokenizer import CharTokenizer


def test_incremental_matches_batch():
    tok = CharTokenizer()
    ids = tok.encode("hello world")
    streamer = TokenStreamer(tok, iter(ids))
    collected = "".join(new for _, new in streamer)
    assert collected == "hello world"


def test_collect_returns_full_text():
    tok = CharTokenizer()
    ids = tok.encode("abc def")
    streamer = TokenStreamer(tok, iter(ids))
    assert streamer.collect() == "abc def"


def test_roundtrip_encode_decode():
    tok = CharTokenizer()
    s = "Hello, world!"
    assert tok.decode(tok.encode(s)) == s
