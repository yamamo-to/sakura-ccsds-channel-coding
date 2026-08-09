"""Additional property‑based tests for the CCSDS Turbo codec implementation.

These tests verify that the interleaver is a permutation, that its inverse works,
and that the puncturing length calculation follows the CCSDS formula.
"""

import pytest
from ccsds_codec.turbo import (
    ccsds_interleaver,
    ccsds_deinterleaver,
    payload_len_from_punctured,
    _puncture,
    _depuncture,
)


def test_interleaver_is_permutation():
    bits = list(range(50))
    inter = ccsds_interleaver(bits)
    # After interleaving, sorted output should equal original sorted bits
    assert sorted(inter) == sorted(bits)
    # No duplicates
    assert len(set(inter)) == len(bits)


def test_deinterleaver_is_inverse():
    bits = list(range(73))  # arbitrary length
    assert ccsds_deinterleaver(ccsds_interleaver(bits)) == bits
    assert ccsds_interleaver(ccsds_deinterleaver(bits)) == bits


@pytest.mark.parametrize("L", [1, 2, 3, 4, 5, 10, 27, 50, 123])
def test_payload_len_from_punctured_roundtrip(L):
    # Build a full (rate‑1/3) stream of length 3*L, then puncture
    full = list(range(3 * L))
    punctured = _puncture(full)
    recovered = payload_len_from_punctured(len(punctured))
    assert recovered == L


def test_depuncture_reconstructs_full_length():
    L = 17
    full = list(range(3 * L))
    punct = _puncture(full)
    recon = _depuncture(punct)
    # The reconstructed stream should have the same systematic and parity1
    # parts; parity2 bits for odd indices are zero.
    assert recon[:L] == full[:L]                     # systematic
    assert recon[L:2 * L] == full[L:2 * L]               # parity1
    # parity2 odd positions should be zero
    for i in range(L):
        expected = full[2 * L + i] if i % 2 == 0 else 0
        assert recon[2 * L + i] == expected
