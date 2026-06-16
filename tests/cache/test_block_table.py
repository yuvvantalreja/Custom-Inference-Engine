import pytest

from engine.cache.block_table import BlockTable

pytestmark = pytest.mark.no_cuda


def test_logical_to_physical():
    t = BlockTable(block_size=4, blocks=[10, 11, 12], length=9)
    assert t.logical_to_physical(0) == (10, 0)
    assert t.logical_to_physical(3) == (10, 3)
    assert t.logical_to_physical(4) == (11, 0)
    assert t.logical_to_physical(8) == (12, 0)


def test_clone_independent():
    t = BlockTable(block_size=2, blocks=[1, 2], length=4)
    c = t.clone()
    c.blocks.append(3)
    c.length = 5
    assert t.blocks == [1, 2]
    assert t.length == 4


def test_truncate_drops_blocks():
    t = BlockTable(block_size=2, blocks=[1, 2, 3], length=6)
    t.truncate(3)
    assert t.length == 3
    assert t.blocks == [1, 2]
