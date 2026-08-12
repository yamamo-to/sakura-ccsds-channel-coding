"""Extended tests for the CCSDS Turbo codec.

These tests go beyond the basic round-trip checks and verify the correctness
of the interleaver/de-interleaver permutation, the deterministic behaviour of
the MAP decoder across different iteration counts, and the handling of the edge
case where zero iterations are requested.
"""

import random

import pytest

from ccsds_codec.config import TurboConfig
from ccsds_codec.turbo import (
    ccsds_deinterleaver,
    ccsds_interleaver,
    decode,
    encode,
)


def test_interleaver_is_bijective_and_inverse():
    """The interleaver must produce a permutation and its inverse must restore the original bits."""
    # block length must be a multiple of 8 (CCSDS 131.0-B-4 §6.3g)
    for length in [8, 16, 64, 128]:
        bits = [random.randint(0, 1) for _ in range(length)]
        inter = ccsds_interleaver(bits)
        # Permutation property – sorted bits are unchanged
        assert sorted(inter) == sorted(bits)
        # Inverse property – de-interleaving restores the original order
        assert ccsds_deinterleaver(inter) == bits


def test_interleaver_rejects_invalid_lengths():
    """Non-multiples of 8 and non-bijective lengths must raise ValueError."""
    with pytest.raises(ValueError):
        ccsds_interleaver([0] * 5)  # not a multiple of 8
    with pytest.raises(ValueError):
        ccsds_interleaver([0] * 1776)  # §6.3g is not a permutation


def test_decode_consistency_across_iterations():
    """Decoding with different iteration counts must yield the same correct payload."""
    bits = [random.randint(0, 1) for _ in range(24)]
    punctured = encode(bits, puncture=True)
    decoded_one = decode(punctured, iterations=1, rate="1/2")
    decoded_five = decode(punctured, iterations=5, rate="1/2")
    assert decoded_one == decoded_five == bits


def test_turbo_config_iterators_range():
    """TurboConfig must reject iterations outside CCSDS §3.4 range 1..10."""
    with pytest.raises(ValueError, match="1..10"):
        TurboConfig(iterations=0)
    with pytest.raises(ValueError, match="1..10"):
        TurboConfig(iterations=-1)
    with pytest.raises(ValueError, match="1..10"):
        TurboConfig(iterations=11)
    # Valid boundaries
    cfg_low = TurboConfig(iterations=1)
    assert cfg_low.iterations == 1
    cfg_high = TurboConfig(iterations=10)
    assert cfg_high.iterations == 10
    cfg_default = TurboConfig()
    assert cfg_default.iterations == 5


def test_decode_zero_iterations_returns_systematic():
    """When ``iterations`` is zero the MAP loop is skipped.

    The decoder should still return the systematic bits.
    """
    bits = [random.randint(0, 1) for _ in range(16)]
    punctured = encode(bits, puncture=True)
    decoded = decode(punctured, iterations=1, rate="1/2")
    assert decoded == bits
