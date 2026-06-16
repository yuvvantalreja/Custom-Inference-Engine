import pytest

pytestmark = pytest.mark.no_cuda


def test_import_version():
    import engine
    assert engine.__version__ == "0.1.0"


def test_import_config():
    from engine import ModelConfig, EngineConfig, SpecConfig, QuantConfig  # noqa
    cfg = ModelConfig(vocab_size=32, hidden_size=16, num_layers=2, num_heads=2)
    assert cfg.head_dim == 8
    assert cfg.num_kv_heads == 2
    assert cfg.intermediate_size == 64
