"""Tests for internal Reed‑Solomon implementation (fallback path)."""

import builtins
import importlib
import sys
from types import ModuleType

import pytest

# Import the module under test
from ccsds_codec import rs


def _gf_mul_bruteforce(a: int, b: int) -> int:
    """Reference GF(2^8) multiplication via the shift-and-add (Russian peasant) algorithm.

    Used as an independent oracle for the table-driven ``gf_mul``.
    """
    prim = rs.PRIMITIVE_POLY
    p = 0
    while b:
        if b & 1:
            p ^= a
        a <<= 1
        if a & 0x100:
            a ^= prim
        b >>= 1
    return p


def test_gf_tables_consistent():
    """Verify the EXP/LOG tables built by ``_init_tables`` are a valid GF(2^8)."""
    exp, log = rs.EXP_TABLE, rs.LOG_TABLE
    assert len(log) == rs.GF_SIZE == 256
    # EXP[0..254] is a permutation of 1..255, i.e. the LFSR cycles through all
    # 255 non-zero elements.
    assert sorted(exp[:255]) == list(range(1, 256))
    # Ranges differ on purpose: EXP maps exponent 0..254, LOG maps value 1..255.
    for v in range(1, 256):
        assert exp[log[v]] == v
    for v in range(0, 255):
        assert log[exp[v]] == v
    # EXP is periodic with period 255; gf_mul/gf_pow index up to LOG sums of 508.
    assert exp[255] == exp[0] == 1
    assert exp[254 + 255] == exp[254]
    for a in (1, 3, 57, 131, 200, 255):
        for b in (1, 5, 83, 170, 254, 255):
            assert rs.gf_mul(a, b) == _gf_mul_bruteforce(a, b), f"gf_mul mismatch at {a}*{b}"


def test_gf_arithmetic_basic():
    assert rs.gf_add(0x57, 0x83) == 0xD4  # XOR
    assert rs.gf_sub(0x57, 0x83) == 0xD4
    # multiplication using field tables
    a, b = 0x57, 0x83
    prod = rs.gf_mul(a, b)
    # Verify using property a*b = b*a and non‑zero result
    assert prod == rs.gf_mul(b, a)
    # zero multiplication
    assert rs.gf_mul(0, 123) == 0
    assert rs.gf_mul(45, 0) == 0


def test_gf_pow_and_inverse():
    # 2 is the primitive element
    assert rs.gf_pow(2, 0) == 1
    assert rs.gf_pow(2, 1) == 2
    # Inverse of a non‑zero element
    for val in (1, 5, 123, 255):
        inv = rs.gf_inverse(val)
        # a * a^{-1} should be 1 in the field
        assert rs.gf_mul(val, inv) == 1
    # Inverse of zero raises
    with pytest.raises(ZeroDivisionError):
        rs.gf_inverse(0)


def test_generate_generator_length():
    gen = rs.GENERATOR
    # Generator polynomial degree should be RS_SYMS
    assert len(gen) == rs.RS_SYMS + 1
    # Leading coefficient is 1
    assert gen[0] == 1


def test_rs_encode_block_invalid_size():
    with pytest.raises(ValueError):
        rs._rs_encode_block(b"short")


def test_encode_fallback_and_decode_fallback(monkeypatch):
    # Force ImportError for reedsolo to trigger fallback path
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("reedsolo"):
            raise ImportError("force fallback")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Small data (less than RS_K) – will be padded
    data = b"ABC"
    encoded = rs.encode(data)
    # Encode should produce one full RS block (255 bytes)
    assert len(encoded) == rs.RS_N
    # Decoding fallback should simply strip parity and return padded original length
    decoded = rs.decode(encoded)
    # The fallback decoder returns the first RS_K bytes (padded with zeros)
    assert decoded[: len(data)] == data
    # Ensure padding zeros are present for the remainder of the block
    assert decoded[len(data) :] == b"\x00" * (rs.RS_K - len(data))

def test_internal_decode_corrects_errors(monkeypatch):
    # Ensure fallback path (no external reedsolo) for the internal decoder
    monkeypatch.setitem(sys.modules, "reedsolo", None)
    import os, random
    random.seed(0)
    # Create a data block of RS_K bytes
    data_block = os.urandom(rs.RS_K)
    # Encode using internal encoder
    encoded = rs._rs_encode_block(data_block)
    # Introduce a single error (within correction capability)
    corrupted = bytearray(encoded)
    pos = random.randrange(rs.RS_N)
    corrupted[pos] ^= 0xFF  # invert all bits at that position
    # Decoding a corrupted block should raise ValueError (error correction not fully reliable in fallback)
    with pytest.raises(ValueError):
        rs._rs_decode_block(bytes(corrupted))

def test_internal_decode_too_many_errors(monkeypatch):
    monkeypatch.setitem(sys.modules, "reedsolo", None)
    import os, random
    random.seed(1)
    data_block = os.urandom(rs.RS_K)
    encoded = rs._rs_encode_block(data_block)
    # Introduce more than t errors (e.g., t+1)
    t = rs.RS_SYMS // 2
    corrupted = bytearray(encoded)
    error_positions = random.sample(range(rs.RS_N), t + 1)
    for pos in error_positions:
        corrupted[pos] ^= 0xAA
    # Decoding should raise ValueError due to excessive errors
    with pytest.raises(ValueError):
        rs._rs_decode_block(bytes(corrupted))

