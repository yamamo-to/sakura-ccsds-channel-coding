"""Tests for utils module"""

import os
import pytest
from ccsds_codec.utils import bytes_to_bits, bits_to_bytes

@pytest.mark.parametrize("data", [b"", b"\x00", b"\xff", b"hello", os.urandom(10)])
def test_bytes_bits_roundtrip(data: bytes):
    bits = bytes_to_bits(data)
    rebuilt = bits_to_bytes(bits)
    assert rebuilt[: len(data)] == data
    assert len(rebuilt) >= len(data)

def test_bits_to_bytes_padding():
    bits = [1, 0, 1, 1, 0]  # 5 bits
    b = bits_to_bytes(bits)
    assert len(b) == 1
    expected = int("10110000", 2).to_bytes(1, "big")
    assert b == expected
