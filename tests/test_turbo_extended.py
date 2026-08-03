"""Extended tests for the CCSDS Turbo codec.

These tests go beyond the basic round‑trip checks and verify the correctness
of the interleaver/de‑interleaver permutation, the deterministic behaviour of
the MAP decoder across different iteration counts, and the handling of the edge
case where zero iterations are requested.
"""

import random

from ccsds_codec.turbo import (
    ccsds_deinterleaver,
    ccsds_interleaver,
    decode,
    encode,
)


def test_interleaver_is_bijective_and_inverse():
    """The interleaver must produce a permutation and its inverse must restore the original bits."""
    for length in [1, 2, 5, 10, 31, 64, 100]:
        bits = [random.randint(0, 1) for _ in range(length)]
        inter = ccsds_interleaver(bits)
        # Permutation property – sorted bits are unchanged
        assert sorted(inter) == sorted(bits)
        # Inverse property – de‑interleaving restores the original order
        assert ccsds_deinterleaver(inter) == bits


def test_decode_consistency_across_iterations():
    """Decoding with different iteration counts must yield the same correct payload."""
    bits = [random.randint(0, 1) for _ in range(20)]
    punctured = encode(bits, puncture=True)
    decoded_one = decode(punctured, iterations=1)
    decoded_five = decode(punctured, iterations=5)
    assert decoded_one == decoded_five == bits


def test_decode_zero_iterations_returns_systematic():
    """When ``iterations`` is zero the MAP loop is skipped – the decoder should still return the systematic bits."""
    bits = [random.randint(0, 1) for _ in range(15)]
    punctured = encode(bits, puncture=True)
    decoded = decode(punctured, iterations=0)
    assert decoded == bits
