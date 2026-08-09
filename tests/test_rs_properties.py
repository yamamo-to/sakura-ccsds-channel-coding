"""Additional property‑based tests for the CCSDS RS(255,223) codec.

These tests check the generator polynomial characteristics, block handling, and
error correction when the external ``reedsolo`` library is available.
"""

import os

import pytest

from ccsds_codec.rs import (
    encode,
    decode,
    RS_K,
    RS_N,
    RS_SYMS,
    GENERATOR,
)

try:
    import reedsolo  # noqa: F401

    _REEDSOLO_AVAILABLE = True
except ImportError:
    _REEDSOLO_AVAILABLE = False


def test_generator_properties():
    # The generator must be monic and have degree RS_SYMS (32) → length 33
    assert len(GENERATOR) == RS_SYMS + 1
    assert GENERATOR[0] == 1  # monic polynomial


def test_encode_block_length():
    data = os.urandom(100)  # less than RS_K, will be padded internally
    enc = encode(data)
    # Length must be a multiple of RS_N (255)
    assert len(enc) % RS_N == 0
    # Number of blocks = ceil(len(data)/RS_K)
    expected_blocks = (len(data) + RS_K - 1) // RS_K
    assert len(enc) == expected_blocks * RS_N


def test_roundtrip_multiple_blocks():
    # Create data spanning three RS blocks (including padding)
    data = os.urandom(RS_K * 3 - 10)
    enc = encode(data)
    dec = decode(enc)
    # ``decode`` strips parity; we compare the original (unpadded part)
    assert dec[: len(data)] == data


def test_fallback_strip_parity_when_no_errors():
    # Ensure that when no external decoder is present we simply strip parity.
    data = b'ABCDEF' * 40  # 240 bytes < RS_K
    enc = encode(data)
    # Corrupt a parity byte – fallback still returns original data (strip only)
    corrupted = bytearray(enc)
    corrupted[-1] ^= 0xFF
    dec = decode(bytes(corrupted))
    assert dec[: len(data)] == data


@pytest.mark.skipif(
    not _REEDSOLO_AVAILABLE,
    reason="External reedsolo decoder is not installed",
)
def test_error_correction_with_reedsolo():
    # Requires ``reedsolo`` with CCSDS parameters configured in decode().
    data = os.urandom(RS_K)
    enc = encode(data)
    # Introduce up to 16 symbol errors (t = RS_SYMS//2)
    corrupted = bytearray(enc)
    for i in range(0, 2 * (RS_SYMS // 2), 2):
        corrupted[i] ^= 0xFF
    dec = decode(bytes(corrupted))
    assert dec[:RS_K] == data
