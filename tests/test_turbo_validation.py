"""Validation and branch-coverage tests for the CCSDS Turbo codec.

These tests pin down the input-validation and edge-case behaviour of
:mod:`ccsds_codec.turbo` (the backwards-compatible shim over
:mod:`ccsds_codec.core.turbo`): invalid bit values, unsupported rates,
non-multiple stream lengths and the byte-padded rate-1/6 helper.  They
also add the first end-to-end round-trip coverage of the rate-1/4 code
(CCSDS 131.0-B-4 §3.4.2 stream layout ``[s_c, uG2_c, uG3_c, lG1_c]``).
"""

import pytest

from ccsds_codec.turbo import (
    decode,
    decode_padded_rate16,
    decode_unpunctured,
    encode,
)


def test_encode_rejects_non_binary_bit():
    """A bit value other than 0/1 must raise ValueError (core/turbo.py:172)."""
    with pytest.raises(ValueError, match="not 0 or 1"):
        encode([0, 1, 2])


def test_encode_rejects_unsupported_rate():
    """An unknown rate string must raise ValueError (core/turbo.py:176)."""
    with pytest.raises(ValueError, match="Unsupported Turbo code rate"):
        encode([0, 1], rate="9/9")


@pytest.mark.parametrize("length", [8, 16, 32])
def test_rate14_roundtrip(length):
    """Rate-1/4 encode/decode round-trips (covers the rate-1/4 demux branch)."""
    bits = [(i * 7 + 3) % 2 for i in range(length)]
    enc = encode(bits, rate="1/4")
    assert len(enc) == 4 * (length + 4)
    dec = decode(enc, rate="1/4", iterations=3)
    assert dec == bits


def test_decode_rejects_stream_not_multiple_of_rate():
    """A rate-1/2 stream of length not a multiple of 2 must raise."""
    # 11 bits: long enough for a frame but 11 % ncomp(2) == 1 (core/turbo.py:441)
    with pytest.raises(ValueError, match="not a multiple of 2"):
        decode([0] * 11, rate="1/2")


def test_decode_rejects_unsupported_rate():
    """An unknown rate passed to decode must raise ValueError (core/turbo.py:498)."""
    with pytest.raises(ValueError, match="Unsupported Turbo code rate"):
        decode([0, 0], rate="bogus")


def test_decode_unpunctured_rejects_unknown_length():
    """A rate-1/3 stream with no valid length in the last 8 bits must raise."""
    with pytest.raises(ValueError, match="no valid length"):
        decode_unpunctured([0] * 13)


def test_decode_padded_rate16_roundtrip_with_padding():
    """Byte-padded rate-1/6 streams decode back to the payload (1..7 pad bits)."""
    bits = [(i * 5) % 2 for i in range(32)]
    enc = encode(bits, rate="1/6")
    assert len(enc) % 8 == 0  # 6*(K+4) is a multiple of 8 when K is
    for pad in range(1, 8):
        padded = enc + [0] * pad
        assert decode_padded_rate16(padded) == bits


def test_decode_padded_rate16_rejects_unknown_length():
    """Garbage too short to contain a rate-1/6 frame must raise."""
    with pytest.raises(ValueError, match="no valid length"):
        decode_padded_rate16([0] * 20)
