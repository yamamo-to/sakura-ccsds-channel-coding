"""Extended tests for Reed‑Solomon encoder/decoder"""

import os
import pytest
from ccsds_codec.rs import encode, decode, RS_K, RS_N, GENERATOR


def test_generator_properties():
    # Generator polynomial degree must be RS_SYMS (32) → length 33
    assert len(GENERATOR) == 33
    # First coefficient should be 1 (monic polynomial)
    assert GENERATOR[0] == 1


@pytest.mark.parametrize("size", [1, 10, RS_K, RS_K + 5, RS_K * 2])
def test_encode_decode_various_lengths(size):
    data = os.urandom(size)
    enc = encode(data)
    # Encoded length must be a multiple of RS_N (255)
    assert len(enc) % RS_N == 0
    dec = decode(enc)
    # decode strips parity; original data may have been padded to RS_K
    assert dec[: len(data)] == data


def test_decode_without_errors_fallback(monkeypatch):
    """Force the fallback path by ensuring reedsolo is not importable.

    The test uses clean data (no corruption) and verifies that the fallback decoder
    correctly returns the original payload when no errors are present.
    """
    import builtins
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("reedsolo"):
            raise ImportError("force fallback")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    data = b"XYZ" * 70  # 210 bytes (< RS_K)
    enc = encode(data)
    # No corruption; fallback should simply strip parity and return original data.
    dec = decode(enc)
    assert dec[: len(data)] == data
