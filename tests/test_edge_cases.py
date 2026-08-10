'''Additional edge‑case tests to improve coverage.'''

import os
import random

from ccsds_codec.utils import bits_to_bytes, bytes_to_bits
from ccsds_codec.conv import encode as conv_encode, viterbi_decode as conv_decode
from ccsds_codec.rs import encode as rs_encode, decode as rs_decode, RS_K
from ccsds_codec.api import ConvCodec, TurboCodec
from ccsds_codec.config import ConvConfig, TurboConfig


def test_bits_to_bytes_empty():
    """bits_to_bytes should return empty bytes for an empty bit list."""
    assert bits_to_bytes([]) == b""


def test_bytes_to_bits_empty():
    """bytes_to_bits should return an empty list for empty bytes."""
    assert bytes_to_bits(b"") == []


def test_conv_encode_decode_empty():
    """Empty input should round‑trip through convolutional codec without error."""
    bits = []
    encoded = conv_encode(bits)
    decoded = conv_decode(encoded)
    assert encoded == []
    assert decoded == []


def test_rs_decode_with_correctable_errors():
    """Introduce a few correctable errors and verify RS decode recovers the data.

    The RS(255,223) code can correct up to 16 symbol errors (t = RS_SYMS // 2).
    """
    data = os.urandom(RS_K)  # exactly one full block
    encoded = rs_encode(data)
    # Flip up to 10 bytes (well within correction capability)
    corrupted = bytearray(encoded)
    for i in range(10):
        corrupted[i] ^= 0xFF
    decoded = rs_decode(bytes(corrupted))
    # Decoder returns only the data portion; original data may have been padded
    assert decoded[: len(data)] == data


def test_rs_decode_exceeds_error_capacity():
    """Introduce more errors than the RS code can correct and expect a failure.

    The external ``reedsolo`` decoder raises ``reedsolo.ReedSolomonError`` when
    uncorrectable.  We accept any exception type here.
    """
    data = os.urandom(RS_K)
    encoded = rs_encode(data)
    corrupted = bytearray(encoded)
    # Introduce 30 errors (> 16)
    for i in range(30):
        corrupted[i] ^= 0xFF
    # Decoding with too many errors should not produce the original data
    decoded = rs_decode(bytes(corrupted))
    assert decoded[: len(data)] != data


def test_conv_codec_rate_passthrough():
    bits = [random.randint(0, 1) for _ in range(64)]
    for rate in ["2/3", "3/4", "5/6", "7/8"]:
        codec = ConvCodec(ConvConfig(rate=rate))
        assert codec.decode(codec.encode(bits)) == bits


def test_turbo_codec_rate16_passthrough():
    bits = [random.randint(0, 1) for _ in range(64)]
    codec = TurboCodec(TurboConfig(rate="1/6"))
    encoded = codec.encode(bits)
    assert len(encoded) == 6 * (64 + 4)
    assert codec.decode(encoded) == bits
