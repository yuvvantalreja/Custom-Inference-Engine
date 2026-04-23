from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BlockTable:
    """Per-sequence logical -> physical block mapping.

    ``block_size`` tokens per block. ``length`` is the number of written tokens.
    ``blocks`` lists physical block ids in order. Supports cheap clone/fork for
    speculative rollback: the clone shares no physical state with the source
    (the caller — typically ``PagedKVCache`` — copies the actual K/V data).
    """

    block_size: int
    blocks: list[int] = field(default_factory=list)
    length: int = 0

    def num_blocks_needed(self, new_length: int) -> int:
        return (new_length + self.block_size - 1) // self.block_size

    def logical_to_physical(self, pos: int) -> tuple[int, int]:
        block_idx = pos // self.block_size
        offset = pos % self.block_size
        return self.blocks[block_idx], offset

    def clone(self) -> "BlockTable":
        return BlockTable(block_size=self.block_size, blocks=list(self.blocks), length=self.length)

    def truncate(self, new_length: int) -> None:
        """Roll back to ``new_length`` tokens, releasing no blocks (caller frees)."""
        assert new_length <= self.length
        self.length = new_length
        needed = self.num_blocks_needed(new_length)
        del self.blocks[needed:]
