from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    intermediate_size: Optional[int] = None
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.head_dim is None:
            assert self.hidden_size % self.num_heads == 0
            self.head_dim = self.hidden_size // self.num_heads
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size


@dataclass
class QuantConfig:
    group_size: int = 128
    symmetric: bool = True
    per_channel: bool = True


@dataclass
class EngineConfig:
    max_batch_size: int = 1
    max_seq_len: int = 2048
    kv_block_size: int = 16
    num_kv_blocks: int = 1024
    device: str = "cuda"
    dtype: str = "float16"  # activation dtype

    def __post_init__(self) -> None:
        if not str(self.device).startswith("cuda"):
            raise ValueError(
                f"GPU-only engine: device must be a CUDA device, got {self.device!r}"
            )


@dataclass
class SpecConfig:
    gamma: int = 4  # draft lookahead
    enabled: bool = False
    draft_config: Optional[ModelConfig] = None
