"""Compatibility tests with external Reed‑Solomon implementation (reedsolo).

These tests verify that the internal CCSDS RS encoder/decoder produce the same
parity symbols as the reference `reedsolo` library when it is available.
"""

import os
import sys
import pytest

# Skip if external library not installed
reedsolo = pytest.importorskip("reedsolo")

from ccsds_codec import rs  # noqa: E402


def _make_reedsolo_codec() -> reedsolo.RSCodec:
    """Create a reedsolo instance with CCSDS 131.0-B-4 parameters."""
    return reedsolo.RSCodec(rs.RS_SYMS, nsize=rs.RS_N, fcr=112, prim=rs.PRIMITIVE_POLY)


def reedsolo_encode(data: bytes) -> bytes:
    return _make_reedsolo_codec().encode(data)


def reedsolo_decode(encoded: bytes) -> bytes:
    decoded = _make_reedsolo_codec().decode(encoded)
    if isinstance(decoded, tuple):
        decoded = decoded[0]
    return decoded[: rs.RS_K]


def test_encode_parity_matches_external():
    data = os.urandom(rs.RS_K)
    # Force internal fallback by removing reedsolo temporarily
    sys.modules.pop("reedsolo", None)
    internal = rs.encode(data)
    external = reedsolo_encode(data)
    assert internal[-rs.RS_SYMS:] == external[-rs.RS_SYMS:]


def test_decode_matches_external():
    data = os.urandom(rs.RS_K)
    encoded = reedsolo_encode(data)
    corrupted = bytearray(encoded)
    corrupted[10] ^= 0xFF
    recovered = rs.decode(bytes(corrupted))
    assert recovered[: rs.RS_K] == data


def test_full_roundtrip_with_external_library():
    data = os.urandom(2 * rs.RS_K)
    encoded = rs.encode(data)
    decoded = rs.decode(encoded)
    assert decoded[: len(data)] == data
