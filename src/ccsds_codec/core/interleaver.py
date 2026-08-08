"""CCSDS quadratic-permutation (QPP) interleaver.

Implements the interleaver described in docs/CCSDS_Turbo_Spec.md §2.2
(CCSDS 131.0-B-4 §3): ``π(i) = (f1·i + f2·i²) mod K``.
"""

from __future__ import annotations

import math

__all__ = ["ccsds_interleaver", "ccsds_deinterleaver", "qpp_perm", "qpp_params"]


def qpp_params(K: int) -> tuple[int, int]:
    """Return valid QPP parameters ``(f1, f2)`` for block length ``K``.

    Generic construction ``f1 = 1``, ``f2 = lcm(rad(K), 4 if 4 | K else 1)``
    satisfies the quadratic-permutation conditions (Sun–Takeshita):
    ``gcd(f1, K) = 1``, every prime divisor of ``K`` divides ``f2``, and
    ``4 | f2`` whenever ``4 | K``.  It is therefore bijective for every
    ``K``, including the CCSDS block lengths 1784/3568/7136/8920/16384.
    """
    rad = 1
    m = K
    d = 2
    while d * d <= m:
        if m % d == 0:
            rad *= d
            while m % d == 0:
                m //= d
        d += 1
    if m > 1:
        rad *= m
    f2 = math.lcm(rad, 4) if K % 4 == 0 else rad
    return 1, f2


def qpp_perm(K: int) -> list[int]:
    """Permutation indices ``π(i) = (f1·i + f2·i²) mod K`` for i = 0..K-1."""
    if K <= 0:
        return []
    f1, f2 = qpp_params(K)
    return [(f1 * i + f2 * i * i) % K for i in range(K)]


def ccsds_interleaver(bits: list[int]) -> list[int]:
    """Apply the CCSDS quadratic-permutation interleaver.

    Output position ``π(i)`` receives input bit ``i`` (``out[π(i)] = bits[i]``,
    docs/CCSDS_Turbo_Spec.md §2.2).  The mapping is a permutation, so
    ``ccsds_deinterleaver(ccsds_interleaver(bits)) == bits``.  It is **not**
    self-inverse.
    """
    K = len(bits)
    if K < 2:
        return bits[:]
    out: list[int] = [0] * K
    f1, f2 = qpp_params(K)
    for i in range(K):
        out[(f1 * i + f2 * i * i) % K] = bits[i]
    return out


def ccsds_deinterleaver(bits: list[int]) -> list[int]:
    """Inverse of :func:`ccsds_interleaver` (``out[i] = bits[π(i)]``)."""
    K = len(bits)
    if K < 2:
        return bits[:]
    out: list[int] = [0] * K
    f1, f2 = qpp_params(K)
    for i in range(K):
        out[i] = bits[(f1 * i + f2 * i * i) % K]
    return out
