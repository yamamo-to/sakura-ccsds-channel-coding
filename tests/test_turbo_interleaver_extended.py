"""Extended CCSDS §6.3g interleaver verification for all standard block lengths.

Since public pre-computed interleaver tables for K = 3568/7136/8920/16384 are
not available as downloadable text files, this module cross-checks the
implementation in ``core.interleaver`` against an independent literal
re-implementation of the CCSDS 131.0-B-4 §6.3g quadratic-permutation algorithm.
"""

from __future__ import annotations

import pytest

from ccsds_codec.core.interleaver import (
    PRIMES,
    ccsds_deinterleaver,
    ccsds_interleaver,
    ccsds_perm,
)

STANDARD_K = (1784, 3568, 7136, 8920, 16384)


def _reference_perm(K: int) -> list[int]:
    """Independent reference implementation of CCSDS §6.3g.

    Returns a 0-based permutation ``perm`` such that
    ``interleaved[j] = bits[perm[j]]``.
    """
    k1 = 8
    if K <= 0 or K % k1 != 0:
        raise ValueError(f"Block length must be a positive multiple of {k1}")
    k2 = K // k1
    perm: list[int] = []
    for s in range(1, K + 1):
        m = (s - 1) % 2
        i = (s - 1) // (2 * k2)
        j = (s - 1) // 2 - i * k2
        t = (19 * i + 1) % (k1 // 2)
        q = t % 8 + 1
        c = (PRIMES[q - 1] * j + 21 * m) % k2
        perm.append(2 * (t + c * (k1 // 2) + 1) - m - 1)
    return perm


@pytest.mark.parametrize("K", STANDARD_K)
def test_ccsds_perm_matches_independent_reference(K: int) -> None:
    """The implementation must match an independent formula-based reference."""
    expected = _reference_perm(K)
    actual = ccsds_perm(K)
    assert actual == expected


@pytest.mark.parametrize("K", STANDARD_K)
def test_ccsds_perm_is_bijection(K: int) -> None:
    """``ccsds_perm`` must return each index exactly once."""
    perm = ccsds_perm(K)
    assert len(perm) == K
    assert set(perm) == set(range(K))
    assert len(set(perm)) == K


@pytest.mark.parametrize("K", STANDARD_K)
def test_ccsds_interleaver_roundtrip(K: int) -> None:
    """Interleaving followed by de-interleaving must recover the input."""
    bits = [(i * 7 + 3) % 2 for i in range(K)]
    interleaved = ccsds_interleaver(bits)
    assert len(interleaved) == K
    recovered = ccsds_deinterleaver(interleaved)
    assert recovered == bits


@pytest.mark.parametrize("K", STANDARD_K)
def test_ccsds_interleaver_follows_permutation(K: int) -> None:
    """``ccsds_interleaver`` must permute bits exactly as ``ccsds_perm`` defines."""
    bits = [(i * 7 + 3) % 2 for i in range(K)]
    perm = ccsds_perm(K)
    expected = [bits[p] for p in perm]
    assert ccsds_interleaver(bits) == expected
