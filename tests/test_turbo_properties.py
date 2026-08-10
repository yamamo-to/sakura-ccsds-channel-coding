"""Additional property-based tests for the CCSDS Turbo codec implementation.

These tests verify that the interleaver is a permutation, that its inverse works,
that the encoded stream length follows the CCSDS formula ``NCOMP[rate] * (K + 4)``,
and that the length-based rate auto-detection is unambiguous.
"""

import pytest

from ccsds_codec.turbo import (
    NCOMP,
    STANDARD_K,
    ccsds_deinterleaver,
    ccsds_interleaver,
    encode,
    _detect_rate_k,
)


def test_interleaver_is_permutation():
    bits = list(range(200))  # multiple of 8 with a bijective §6.3g permutation
    inter = ccsds_interleaver(bits)
    # After interleaving, sorted output should equal original sorted bits
    assert sorted(inter) == sorted(bits)
    # No duplicates
    assert len(set(inter)) == len(bits)


def test_deinterleaver_is_inverse():
    bits = list(range(200))
    assert ccsds_deinterleaver(ccsds_interleaver(bits)) == bits
    assert ccsds_interleaver(ccsds_deinterleaver(bits)) == bits


@pytest.mark.parametrize("rate", ["1/2", "1/3", "1/4", "1/6"])
def test_stream_length_formula(rate):
    # each rate produces NCOMP[rate] * (K + 4) bits (TAIL = 4 termination)
    K = 64
    enc = encode([0] * K, rate=rate)
    assert len(enc) == NCOMP[rate] * (K + 4)


def test_rate_autodetect_is_unique_and_correct():
    # all 20 standard (rate, K) pairs map to distinct stream lengths
    lengths = {NCOMP[r] * (K + 4) for r in NCOMP for K in STANDARD_K}
    assert len(lengths) == len(NCOMP) * len(STANDARD_K)
    for rate in NCOMP:
        for K in STANDARD_K:
            stream_len = NCOMP[rate] * (K + 4)
            assert _detect_rate_k(stream_len) == (rate, K)


def test_rate_autodetect_rejects_unknown_length():
    with pytest.raises(ValueError):
        _detect_rate_k(12345)
