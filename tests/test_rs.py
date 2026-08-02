"""Tests for Reed‑Solomon encoder/decoder"""

import os
import pytest
from ccsds_codec.rs import encode, decode, RS_K, RS_N, GENERATOR

def test_generator_length():
    # Generator polynomial should have degree RS_SYMS (32) → length 33
    assert len(GENERATOR) == 33

def test_encode_decode_roundtrip():
    # Create data of arbitrary length (not necessarily multiple of RS_K)
    data = os.urandom(100)  # 100 < RS_K
    enc = encode(data)
    # Length should be a multiple of RS_N (255)
    assert len(enc) % RS_N == 0
    dec = decode(enc)
    # Decoder strips parity (fallback) → should recover original data (padded zeros trimmed)
    assert dec[: len(data)] == data
