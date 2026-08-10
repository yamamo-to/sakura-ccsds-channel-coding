"""Tests for RS encoder interleaving depth (I=1..5)."""

import os
import pytest

from ccsds_codec.rs import encode, encode_block, RS_K, RS_SYMS

def test_depth_one_compatibility():
    for size in [0, 1, 10, 223, 500, 1000]:
        data = os.urandom(size)
        assert encode(data, 1) == encode(data)

@pytest.mark.parametrize("depth", [2, 3, 4, 5])
def test_interleaving_structure(depth):
    data_len = RS_K * depth - 10
    data = os.urandom(data_len)
    enc = encode(data, depth)
    padded = data.ljust(RS_K * depth, b"\x00")
    assert enc[: RS_K * depth] == padded
    for j in (0, RS_SYMS - 1):
        for i in range(depth):
            block = padded[i::depth]
            expected = encode_block(block)
            parity_byte = expected[RS_K + j]
            offset = RS_K * depth + j * depth + i
            assert enc[offset] == parity_byte

def test_empty_input():
    assert encode(b"", 2) == b""

def test_invalid_depth():
    data = os.urandom(10)
    with pytest.raises(ValueError):
        encode(data, 0)
    with pytest.raises(ValueError):
        encode(data, 6)
