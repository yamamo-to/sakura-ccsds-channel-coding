"""Tests for RS encoder/decoder interleaving depth (I=1..5)."""

import os
import pytest

from ccsds_codec.rs import decode, encode, encode_block, RS_K, RS_SYMS


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


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_roundtrip(depth):
    for size in [1, 100, RS_K * depth, RS_K * depth + 17, 2 * RS_K * depth]:
        data = os.urandom(size)
        dec = decode(encode(data, depth), depth)
        assert dec[:len(data)] == data


def test_depth_one_decode_compatibility():
    for size in [0, 1, 10, 223, 500, 1000]:
        data = os.urandom(size)
        assert decode(encode(data), 1) == decode(encode(data))


def test_decode_malformed_length():
    with pytest.raises(ValueError):
        decode(b"\x00" * 254, 2)  # 254 % (255 * 2) != 0


def test_decode_mismatched_depth_does_not_recover():
    """When decoding with a mismatched interleaving depth, the output must not equal the original.

    The decoder may raise ``ValueError`` due to length checks, or it may return a
    mismatched payload. Both outcomes are acceptable; the original data must never be recovered.
    """
    data = os.urandom(RS_K * 3)
    try:
        decoded = decode(encode(data, 3), 1)
    except ValueError:
        pass
    else:
        assert decoded != data
    try:
        decoded = decode(encode(data, 1), 3)
    except ValueError:
        pass
    else:
        assert decoded != data


def test_decode_empty_input():
    assert decode(b"", 2) == b""
