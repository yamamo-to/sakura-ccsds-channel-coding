"""Tests for Reed‑Solomon encoder/decoder"""

import os

import pytest

from ccsds_codec.rs import encode, decode, RS_K, RS_N, GENERATOR


def test_generator_length():
    # Generator polynomial should have degree RS_SYMS (32) → length 33
    assert len(GENERATOR) == 33


# Golden vector generated with reedsolo using CCSDS 131.0-B-4 parameters
# (fcr=112, prim=0x187) for the payload bytes(range(RS_K)).
_CCSDS_RS_PARITY_RANGE = bytes.fromhex(
    "9ee74a9b27f43ace1a8d80fcffa18456c47eea805a5aa07d62914bbabfcbfe51"
)


@pytest.mark.parametrize(
    ("data", "expected_parity"),
    [
        (b"\x00" * RS_K, b"\x00" * 32),
        (bytes(range(RS_K)), _CCSDS_RS_PARITY_RANGE),
    ],
)
def test_encode_known_vector(data: bytes, expected_parity: bytes) -> None:
    """Encoder parity must match the CCSDS reference polynomial."""
    enc = encode(data)
    assert len(enc) == RS_N
    assert enc[RS_K:] == expected_parity


def test_encode_decode_roundtrip():
    # Create data of arbitrary length (not necessarily multiple of RS_K)
    data = os.urandom(100)  # 100 < RS_K
    enc = encode(data)
    # Length should be a multiple of RS_N (255)
    assert len(enc) % RS_N == 0
    dec = decode(enc)
    # Decoder strips parity (fallback) → should recover original data (padded zeros trimmed)
    assert dec[: len(data)] == data
