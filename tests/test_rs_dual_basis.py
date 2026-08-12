"""Tests for RS dual‑basis transformation (CCSDS 131.0‑B‑4 §4.1 note)."""

import os
import random

from ccsds_codec import RSCodec, RSConfig
from ccsds_codec.core.galois import (
    DUAL_BASIS,
    DUAL_TRACE_MULT_COEFFS,
    gf_from_dual_basis,
    gf_to_dual_basis,
)
from ccsds_codec.core.reed_solomon import (
    decode,
    decode_block,
    encode,
    encode_block,
    RS_K,
)


class TestDualBasisConversion:
    """Round‑trip and correctness of the dual‑basis conversion functions."""

    def test_roundtrip_all_256_values(self):
        """gf_from_dual_basis(gf_to_dual_basis(x)) == x for all x in 0..255."""
        for x in range(256):
            assert gf_from_dual_basis(gf_to_dual_basis(x)) == x

    def test_bijection(self):
        """gf_to_dual_basis is a bijection (all 256 images are unique)."""
        images = {gf_to_dual_basis(x) for x in range(256)}
        assert len(images) == 256

    def test_dual_basis_constants_defined(self):
        """DUAL_BASIS and DUAL_TRACE_MULT_COEFFS have length 8."""
        assert len(DUAL_BASIS) == 8
        assert len(DUAL_TRACE_MULT_COEFFS) == 8

    def test_dual_basis_elements_nonzero(self):
        """All dual-basis elements are nonzero field elements."""
        for i, d in enumerate(DUAL_BASIS):
            assert d != 0, f"d_{i} is zero"

    def test_dual_basis_elements_in_gf256(self):
        """All dual-basis elements are valid GF(2⁸) values."""
        for d in DUAL_BASIS:
            assert 0 <= d <= 255

    def test_trace_mult_coeffs_in_range(self):
        """All trace multiplication coefficients are 8-bit values."""
        for t in DUAL_TRACE_MULT_COEFFS:
            assert 0 <= t <= 255


class TestEncodeBlockDualBasis:
    """encode_block with dual_basis=True produces valid dual‑basis codewords."""

    def test_dual_encode_verifies(self):
        """Parity check passes after dual-basis encode."""
        data = bytes(range(RS_K))
        dual_enc = encode_block(data, dual_basis=True)
        # Decode with dual_basis=True uses our own parity check.
        dec = decode_block(dual_enc, dual_basis=True)
        # dec is in dual-basis; convert back
        dec_conv = bytes(gf_from_dual_basis(b) for b in dec)
        assert dec_conv == data

    def test_dual_enc_differs_from_conventional(self):
        """Dual-basis encoding produces different output than conventional."""
        data = bytes(range(RS_K))
        conv_enc = encode_block(data)
        dual_enc = encode_block(data, dual_basis=True)
        # The data portion differs (dual-basis transformed)
        assert dual_enc[:RS_K] != conv_enc[:RS_K]

    def test_conventional_roundtrip_unchanged(self):
        """encode_block without dual_basis still works."""
        data = bytes(range(RS_K))
        enc = encode_block(data)
        dec = decode_block(enc)
        assert dec == data


class TestTopLevelDualBasis:
    """Top-level encode/decode with dual_basis flag."""

    def test_roundtrip_no_interleave(self):
        """encode/decode at depth=1 with dual_basis=True roundtrips."""
        data = os.urandom(RS_K)
        enc = encode(data, depth=1, dual_basis=True)
        dec = decode(enc, depth=1, dual_basis=True)
        # dec is in dual-basis; convert back
        dec_conv = bytes(gf_from_dual_basis(b) for b in dec)
        assert dec_conv == data

    def test_roundtrip_depth_2(self):
        """encode/decode at depth=2 with dual_basis=True roundtrips."""
        data = os.urandom(RS_K * 2)
        enc = encode(data, depth=2, dual_basis=True)
        dec = decode(enc, depth=2, dual_basis=True)
        dec_conv = bytes(gf_from_dual_basis(b) for b in dec)
        assert dec_conv == data

    def test_empty_data(self):
        """encode/decode with empty data and dual_basis=True."""
        assert encode(b"", dual_basis=True) == b""
        assert decode(b"", dual_basis=True) == b""


class TestRSCodecApiDualBasis:
    """RSCodec class respects config.dual_basis."""

    def test_default_config_dual_basis_false(self):
        """Default RSConfig has dual_basis=False."""
        cfg = RSConfig()
        assert cfg.dual_basis is False

    def test_config_dual_basis_true(self):
        """RSConfig(dual_basis=True) sets the flag."""
        cfg = RSConfig(dual_basis=True)
        assert cfg.dual_basis is True

    def test_codec_roundtrip_dual_basis(self):
        """RSCodec with dual_basis=True roundtrips data."""
        cfg = RSConfig(depth=1, dual_basis=True)
        codec = RSCodec(cfg)
        data = os.urandom(RS_K)
        enc = codec.encode(data)
        dec = codec.decode(enc)
        # decode returns dual-basis; convert back
        dec_conv = bytes(gf_from_dual_basis(b) for b in dec)
        assert dec_conv == data

    def test_codec_roundtrip_conventional(self):
        """RSCodec with default config (no dual_basis) still works."""
        codec = RSCodec()
        data = os.urandom(RS_K)
        enc = codec.encode(data)
        dec = codec.decode(enc)
        assert dec == data


class TestDualBasisErrorCorrection:
    """Dual-basis RS supports full error correction via reedsolo."""

    def test_correct_single_symbol_error(self):
        """Correct 1 corrupted symbol with dual-basis encode/decode."""
        random.seed(42)
        data = bytes(random.getrandbits(8) for _ in range(RS_K))
        enc = encode(data, dual_basis=True)
        # Corrupt 1 byte in the dual-basis codeword
        enc_bytes = bytearray(enc)
        enc_bytes[RS_K + 10] ^= 0xFF
        dec = decode(bytes(enc_bytes), dual_basis=True)
        dec_conv = bytes(gf_from_dual_basis(b) for b in dec)
        assert dec_conv == data

    def test_correct_multiple_symbol_errors(self):
        """Correct 10 corrupted symbols (within RS(255,223) t=16 limit)."""
        random.seed(123)
        data = bytes(random.getrandbits(8) for _ in range(RS_K * 2))
        enc = encode(data, dual_basis=True)
        # Corrupt 10 bytes scattered across the codeword
        enc_bytes = bytearray(enc)
        positions = [3, 45, 100, 150, 200, 220, 50, 120, 180, 240]
        for pos in positions:
            enc_bytes[pos] ^= 0xAB
        dec = decode(bytes(enc_bytes), dual_basis=True)
        dec_conv = bytes(gf_from_dual_basis(b) for b in dec)
        assert dec_conv == data

    def test_codec_corrects_errors(self):
        """RSCodec with dual_basis=True corrects errors."""
        random.seed(456)
        cfg = RSConfig(depth=1, dual_basis=True)
        codec = RSCodec(cfg)
        data = bytes(random.getrandbits(8) for _ in range(RS_K))
        enc = codec.encode(data)
        enc_bytes = bytearray(enc)
        enc_bytes[10] ^= 0xFF
        enc_bytes[RS_K + 5] ^= 0x7F
        dec = codec.decode(bytes(enc_bytes))
        dec_conv = bytes(gf_from_dual_basis(b) for b in dec)
        assert dec_conv == data
