from __future__ import annotations

from dataclasses import dataclass

import torch

from engine.cache.block_table import BlockTable


@dataclass
class KVCacheConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    num_blocks: int
    block_size: int
    dtype: torch.dtype = torch.float32
    device: torch.device | str = "cpu"

    def bytes_required(self) -> int:
        """Total VRAM footprint across all layers and K+V."""
        per_elem = torch.tensor([], dtype=self.dtype).element_size()
        return (
            2  # K + V
            * self.num_layers
            * self.num_blocks
            * self.block_size
            * self.num_kv_heads
            * self.head_dim
            * per_elem
        )


class PagedKVCache:
    """Block-based K/V storage shared across sequences.

    Layout per layer: K, V tensors of shape
        (num_blocks, block_size, num_kv_heads, head_dim).
    Sequences own a ``BlockTable`` that maps logical positions to physical
    block ids. ``fork`` copies a sequence's K/V into freshly allocated blocks
    so speculative rejection can roll back without disturbing the parent.
    """

    def __init__(self, cfg: KVCacheConfig):
        self.cfg = cfg
        device = torch.device(cfg.device)
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info()
            need = cfg.bytes_required()
            if need > free:
                raise RuntimeError(
                    f"PagedKVCache would need {need/2**30:.2f} GiB but only "
                    f"{free/2**30:.2f} GiB free on {device}"
                )
        shape = (cfg.num_blocks, cfg.block_size, cfg.num_kv_heads, cfg.head_dim)
        self.k = [torch.zeros(shape, dtype=cfg.dtype, device=device) for _ in range(cfg.num_layers)]
        self.v = [torch.zeros(shape, dtype=cfg.dtype, device=device) for _ in range(cfg.num_layers)]
        self._free: list[int] = list(range(cfg.num_blocks))

    @property
    def device(self) -> torch.device:
        return self.k[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.k[0].dtype

    # --- block allocator ---

    def _alloc_block(self) -> int:
        if not self._free:
            raise RuntimeError("KV cache out of blocks")
        return self._free.pop()

    def _free_block(self, block_id: int) -> None:
        self._free.append(block_id)

    def new_sequence(self) -> BlockTable:
        return BlockTable(block_size=self.cfg.block_size)

    def free_sequence(self, table: BlockTable) -> None:
        for b in table.blocks:
            self._free_block(b)
        table.blocks.clear()
        table.length = 0

    # --- write / read ---

    def _validate(self, t: torch.Tensor) -> None:
        if t.device != self.device or t.dtype != self.dtype:
            raise ValueError(
                f"tensor device/dtype ({t.device}/{t.dtype}) does not match "
                f"cache ({self.device}/{self.dtype})"
            )

    def append(
        self,
        layer: int,
        table: BlockTable,
        k: torch.Tensor,
        v: torch.Tensor,
        start: int | None = None,
    ) -> None:
        """Write k, v (shape (L, num_kv_heads, head_dim)) into the cache.

        Writes span one or two blocks per call in the common case (L <=
        block_size). For long prefills we walk blocks and do one slice
        assignment per block — ``ceil(L/block_size)+1`` copies total, not L.
        """
        L = k.shape[0]
        assert v.shape[0] == L
        self._validate(k)
        self._validate(v)
        if start is None:
            start = table.length
        end = start + L
        needed = table.num_blocks_needed(end)
        while len(table.blocks) < needed:
            table.blocks.append(self._alloc_block())

        bs = self.cfg.block_size
        src_off = 0
        pos = start
        while src_off < L:
            bid, off = table.logical_to_physical(pos)
            take = min(bs - off, L - src_off)
            self.k[layer][bid, off : off + take] = k[src_off : src_off + take]
            self.v[layer][bid, off : off + take] = v[src_off : src_off + take]
            pos += take
            src_off += take

        if end > table.length:
            table.length = end

    def gather(self, layer: int, table: BlockTable) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather the full K/V for a sequence into dense (L, Hk, D) tensors.

        Single ``index_select`` across the block table, then a reshape + slice
        to ``L`` — one allocation and one copy instead of per-block ``cat``.
        """
        L = table.length
        if L == 0:
            shape = (0, self.cfg.num_kv_heads, self.cfg.head_dim)
            empty = self.k[layer].new_zeros(shape)
            return empty, empty.clone()
        idx = torch.tensor(table.blocks, device=self.device, dtype=torch.long)
        k = self.k[layer].index_select(0, idx).reshape(-1, self.cfg.num_kv_heads, self.cfg.head_dim)[:L]
        v = self.v[layer].index_select(0, idx).reshape(-1, self.cfg.num_kv_heads, self.cfg.head_dim)[:L]
        return k, v

    # --- speculative fork/rollback ---

    def fork(self, table: BlockTable) -> BlockTable:
        """Return a new BlockTable whose K/V are a deep copy of ``table``'s.

        One ``index_copy_`` per layer across all source→dest blocks, instead
        of ``num_blocks × num_layers`` separate copies.
        """
        n = len(table.blocks)
        new_blocks = [self._alloc_block() for _ in range(n)]
        if n > 0:
            src = torch.tensor(table.blocks, device=self.device, dtype=torch.long)
            dst = torch.tensor(new_blocks, device=self.device, dtype=torch.long)
            for layer in range(self.cfg.num_layers):
                self.k[layer].index_copy_(0, dst, self.k[layer].index_select(0, src))
                self.v[layer].index_copy_(0, dst, self.v[layer].index_select(0, src))
        return BlockTable(block_size=self.cfg.block_size, blocks=new_blocks, length=table.length)

    def rollback(self, table: BlockTable, new_length: int) -> None:
        """Free blocks made unnecessary by truncating to ``new_length``."""
        assert new_length <= table.length
        needed = table.num_blocks_needed(new_length)
        for bid in table.blocks[needed:]:
            self._free_block(bid)
        table.blocks[needed:] = []
        table.length = new_length
