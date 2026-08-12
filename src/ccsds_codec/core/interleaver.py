"""CCSDS Turbo interleaver (CCSDS 131.0-B-4 §6.3g).

Implements the block interleaver defined by the *Quadratic-Permutation*
construction of CCSDS 131.0-B-4 §6.3g (Annex H of the Turbo coding
recommendation): with ``k1 = 8`` and ``k2 = K / 8`` the output position ``j``
(0-based) is fed by input position ``π(j)``, i.e. ``interleaved[j] = bits[π(j)]``.

The permutation for K = 1784 is golden-vector verified against the CCSDS
reference interleaver table ``ccsdsSize1784.txt`` from the
``mdmoctezuma/CCSDSTurboCode`` repository (see
``tests/test_turbo_golden.py``).  With ``k1 = 8`` fixed (CCSDS 131.0-B-4
Table 6-3) the construction covers every standard block length
1784/3568/7136/8920; K = 16384 is an extension (not a standard Turbo
block length, LDPC §7.4 only).
"""

from __future__ import annotations

__all__ = ["ccsds_interleaver", "ccsds_deinterleaver", "ccsds_perm"]

#: Odd-prime table used by the CCSDS §6.3g construction (k1 = 8).
PRIMES = (31, 37, 43, 47, 53, 59, 61, 67)

_K1 = 8
_perm_cache: dict[int, list[int]] = {}


def ccsds_perm(K: int) -> list[int]:
    """Return the CCSDS §6.3g permutation indices for block length ``K``.

    The result ``perm`` satisfies ``interleaved[j] = bits[perm[j]]`` for
    ``j = 0 .. K-1``.  ``K`` must be divisible by 8 (``k2 = K/8``); the CCSDS
    standard block lengths (1784, 3568, 7136, 8920) and the extension
    16384 all satisfy this.

    Args:
        K: Block length in bits.

    Returns:
        Permutation indices, one per output position.

    Raises:
        ValueError: If ``K`` is not a positive multiple of 8.
    """
    if K <= 0 or K % _K1 != 0:
        raise ValueError(f"CCSDS §6.3g interleaver requires a block length divisible by 8, got {K}")
    if K in _perm_cache:
        return _perm_cache[K]
    k2 = K // _K1
    perm: list[int] = []
    for s in range(1, K + 1):  # 1-based position within the block
        m = (s - 1) % 2
        i = (s - 1) // (2 * k2)
        j = (s - 1) // 2 - i * k2
        t = (19 * i + 1) % (_K1 // 2)
        q = t % 8 + 1
        c = (PRIMES[q - 1] * j + 21 * m) % k2
        perm.append(2 * (t + c * (_K1 // 2) + 1) - m - 1)
    if len(set(perm)) != K:
        raise ValueError(
            f"CCSDS §6.3g interleaver is not a permutation for block length {K}; "
            "the CCSDS standard only guarantees the construction for the block "
            "lengths 1784, 3568, 7136, 8920 and 16384"
        )
    _perm_cache[K] = perm
    return perm


def ccsds_interleaver(bits: list[int]) -> list[int]:
    """Apply the CCSDS §6.3g interleaver: ``out[j] = bits[perm[j]]``.

    Args:
        bits: Input bit list (length a positive multiple of 8).

    Returns:
        Interleaved bit list of the same length.
    """
    perm = ccsds_perm(len(bits))
    return [bits[p] for p in perm]


def ccsds_deinterleaver(bits: list[int]) -> list[int]:
    """Inverse of :func:`ccsds_interleaver`: ``out[perm[j]] = bits[j]``.

    Args:
        bits: Interleaved bit list (length a positive multiple of 8).

    Returns:
        De-interleaved bit list of the same length.
    """
    perm = ccsds_perm(len(bits))
    out: list[int] = [0] * len(bits)
    for j, p in enumerate(perm):
        out[p] = bits[j]
    return out
