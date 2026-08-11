"""Edge-case tests closing coverage gaps in core helpers and the class API."""

import importlib

import pytest

from ccsds_codec.api import Randomizer, TurboCodec
from ccsds_codec.config import TurboConfig
from ccsds_codec.core.bits import bits_to_bytes, bits_to_bytes_strict
from ccsds_codec.core.convolutional import (
    decode as conv_decode,
    decode_byte_padded,
    encode as conv_encode,
    encode_cxx,
)
from ccsds_codec.core.galois import gf_pow
from ccsds_codec.core.reed_solomon import _fallback_decode_block
from ccsds_codec.core.reed_solomon import decode as rs_decode


def test_bits_to_bytes_rejects_invalid_bit_values():
    """bits_to_bytes raises ValueError for values other than 0 or 1."""
    with pytest.raises(ValueError):
        bits_to_bytes([0, 2, 1, 0])


def test_bits_to_bytes_strict_roundtrip():
    """bits_to_bytes_strict packs 8-aligned bits without padding."""
    assert bits_to_bytes_strict([1, 0, 1, 0, 1, 0, 1, 0]) == b"\xaa"


def test_bits_to_bytes_strict_rejects_partial_byte():
    """bits_to_bytes_strict raises ValueError for lengths not a multiple of 8."""
    with pytest.raises(ValueError):
        bits_to_bytes_strict([1, 0, 1, 0])


def test_gf_pow_zero_base_returns_zero():
    """gf_pow(0, n) is 0 for every exponent (alpha^0 would be 1, but 0^0 is 0)."""
    assert gf_pow(0, 0) == 0
    assert gf_pow(0, 5) == 0


def test_rs_fallback_decode_block_rejects_wrong_length():
    """The fallback RS decoder requires blocks of exactly RS_N bytes."""
    with pytest.raises(ValueError, match="exactly 255"):
        _fallback_decode_block(b"\x00" * 10)


def test_rs_decode_rejects_out_of_range_depth():
    """rs_decode rejects interleaving depths outside 1..5."""
    with pytest.raises(ValueError, match="depth"):
        rs_decode(b"\x00" * 255, depth=0)


def test_main_module_imports():
    """The ``python -m ccsds_codec`` entry module imports cleanly."""
    importlib.import_module("ccsds_codec.__main__")


def test_randomizer_api_scramble_descramble():
    """The Randomizer class is stateless and its own inverse."""
    bits = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1]
    assert Randomizer.descramble(Randomizer.scramble(bits)) == bits


def test_turbo_codec_decode_unpunctured():
    """TurboCodec.decode_unpunctured recovers bits from a rate-1/3 stream."""
    codec = TurboCodec(TurboConfig(rate="1/3"))
    bits = [0, 1, 1, 0, 1, 0, 0, 1]  # length must be a multiple of 8 (QPP interleaver)
    decoded = codec.decode_unpunctured(codec.encode(bits))
    assert decoded == bits


def test_conv_encode_rejects_invalid_bit_value():
    """conv_encode raises ValueError when a bit is not 0 or 1."""
    with pytest.raises(ValueError, match="Bit at position"):
        conv_encode([0, 2])


def test_encode_cxx_rejects_empty_input():
    """encode_cxx refuses an empty payload (mirrors the C++ encoder contract)."""
    with pytest.raises(ValueError, match="must not be empty"):
        encode_cxx([])


def test_encode_cxx_without_termination_is_prefix():
    """encode_cxx(terminate=False) equals the head of the flushed stream."""
    bits = [0, 1, 1, 0, 1, 0, 0, 1]
    assert encode_cxx(bits, terminate=False) == encode_cxx(bits, terminate=True)[: 2 * len(bits)]


def test_conv_decode_wrapper_roundtrip():
    """The core decode() wrapper forwards to the Viterbi decoder."""
    bits = [0, 1, 1, 0, 1, 0, 0, 1]
    encoded = conv_encode(bits, rate="1/2")
    assert conv_decode(encoded, rate="1/2") == bits


def test_conv_decode_empty_punctured_stream():
    """An empty punctured stream decodes to nothing (no erasures to reinsert)."""
    assert conv_decode([], rate="2/3") == []


def test_decode_byte_padded_rejects_unrecognized_stream():
    """decode_byte_padded raises when no trailing-pad trim yields a valid byte boundary."""
    with pytest.raises(ValueError, match="no valid length"):
        decode_byte_padded([0, 1, 0, 1, 0, 1, 0, 1, 0], rate="2/3")
