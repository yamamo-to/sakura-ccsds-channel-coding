"""GF(2^8) arithmetic for the CCSDS Reed‑Solomon code.

Implements GF(2^8) using the CCSDS 131.0‑B‑4 field generator polynomial
``p(x) = x^8 + x^7 + x^2 + x + 1`` (``0x187``).

Also provides dual‑basis transformation utilities (CCSDS 131.0‑B‑4 §4.1
note) for Reed‑Solomon encoding/decoding in the dual basis.
"""

from __future__ import annotations

from typing import List

# Field generator polynomial for CCSDS 131.0-B-4 Reed-Solomon.
PRIMITIVE_POLY = 0x187  # x^8 + x^7 + x^2 + x + 1
GF_SIZE = 256
EXP_TABLE: List[int] = [0] * (GF_SIZE * 2)
LOG_TABLE: List[int] = [0] * GF_SIZE


def _init_tables() -> None:
    x = 1
    for i in range(GF_SIZE - 1):
        EXP_TABLE[i] = x
        LOG_TABLE[x] = i
        x <<= 1
        if x & 0x100:
            x ^= PRIMITIVE_POLY
    for i in range(GF_SIZE - 1, GF_SIZE * 2):
        EXP_TABLE[i] = EXP_TABLE[i - (GF_SIZE - 1)]


_init_tables()


def gf_add(a: int, b: int) -> int:
    """GF(2^8) addition (XOR)."""
    return a ^ b


def gf_sub(a: int, b: int) -> int:
    """GF(2^8) subtraction (identical to addition over GF(2^8))."""
    return a ^ b


def gf_mul(a: int, b: int) -> int:
    """GF(2^8) multiplication via log/exp tables."""
    if a == 0 or b == 0:
        return 0
    return EXP_TABLE[LOG_TABLE[a] + LOG_TABLE[b]]


def gf_pow(a: int, power: int) -> int:
    """Raise a field element *a* to the integer *power*."""
    if a == 0:
        return 0
    return EXP_TABLE[(LOG_TABLE[a] * power) % (GF_SIZE - 1)]


def gf_inverse(a: int) -> int:
    """Multiplicative inverse of a non‑zero field element *a*."""
    if a == 0:
        raise ZeroDivisionError("inverse of 0 does not exist")
    return EXP_TABLE[(GF_SIZE - 1) - LOG_TABLE[a]]


# ---------------------------------------------------------------------------
# Dual‑basis transformation (CCSDS 131.0‑B‑4 §4.1 note)
# ---------------------------------------------------------------------------
#
# The conventional basis of GF(2⁸) is ``{1, α, α², …, α⁷}`` where ``α = 2``
# is the primitive element and the field generator is ``p(x) = x⁸ + x⁷ + x²
# + x + 1`` (``0x187``).
#
# The *dual basis* ``{d₀, d₁, …, d₇}`` satisfies
#
#     tr(dᵢ · αʲ) = δᵢⱼ    (Kronecker delta)
#
# where ``tr(x) = x + x² + x⁴ + … + x¹²⁸`` is the absolute trace from
# GF(2⁸) to GF(2).
#
# Dual‑basis coefficients ``cᵢ`` of a field element ``x`` are computed as
#
#     cᵢ = tr(x · dᵢ)
#
# and the element is reconstructed by ``x = ⊕_{cᵢ=1} dᵢ``.
# The precomputed trace multiplication coefficients ``T[i]`` are defined so
# that ``tr(x · dᵢ)`` equals the parity of ``(x & T[i])`` (popcount mod 2)
# when ``x`` is in conventional‑basis representation.
# ---------------------------------------------------------------------------

#: Dual basis elements for the CCSDS ``p(x) = x⁸ + x⁷ + x² + x + 1``.
DUAL_BASIS: List[int] = [0x03, 0xC1, 0xA0, 0x50, 0x28, 0x14, 0x0A, 0x06]

#: Trace multiplication coefficients.
#: ``tr(x · dᵢ) = popcount(x & T[i]) & 1`` for ``x`` in conventional basis.
DUAL_TRACE_MULT_COEFFS: List[int] = [0xFE, 0xFF, 0x7F, 0xBF, 0x5F, 0xAF, 0x57, 0xAB]


def gf_trace(x: int) -> int:
    """Compute the absolute trace ``tr(x) = x + x² + x⁴ + … + x¹²⁸``.

    Args:
        x: Field element (0 .. 255) in conventional‑basis representation.

    Returns:
        ``0`` or ``1`` — the trace value in GF(2).
    """
    t = x
    t ^= gf_mul(t, t)      # t²
    t ^= gf_mul(t, t)      # t⁴
    t ^= gf_mul(t, t)      # t⁸
    return t & 1


def gf_to_dual_basis(value: int) -> int:
    """Convert a field element from conventional‑basis to dual‑basis.

    The *conventional‑basis* representation of a value ``v`` (0 .. 255) uses
    the bits of ``v`` as coefficients of ``{1, α, α², …, α⁷}``.
    The *dual‑basis* representation reuses the same integer container but
    interprets the bits as coefficients of the dual basis ``{d₀, …, d₇}``.

    Args:
        value: Field element in conventional‑basis representation.

    Returns:
        The same field element expressed in dual‑basis representation.
    """
    v = value
    r = 0
    # Unrolled loop: tr(v · dᵢ) = popcount(v & T[i]) & 1
    if (v & DUAL_TRACE_MULT_COEFFS[0]).bit_count() & 1: r |= 1 << 0
    if (v & DUAL_TRACE_MULT_COEFFS[1]).bit_count() & 1: r |= 1 << 1
    if (v & DUAL_TRACE_MULT_COEFFS[2]).bit_count() & 1: r |= 1 << 2
    if (v & DUAL_TRACE_MULT_COEFFS[3]).bit_count() & 1: r |= 1 << 3
    if (v & DUAL_TRACE_MULT_COEFFS[4]).bit_count() & 1: r |= 1 << 4
    if (v & DUAL_TRACE_MULT_COEFFS[5]).bit_count() & 1: r |= 1 << 5
    if (v & DUAL_TRACE_MULT_COEFFS[6]).bit_count() & 1: r |= 1 << 6
    if (v & DUAL_TRACE_MULT_COEFFS[7]).bit_count() & 1: r |= 1 << 7
    return r


def gf_from_dual_basis(value: int) -> int:
    """Convert a field element from dual‑basis to conventional‑basis.

    This is the inverse of :func:`gf_to_dual_basis`.

    Args:
        value: Field element in dual‑basis representation.

    Returns:
        The same field element expressed in conventional‑basis representation.
    """
    r = 0
    # Unrolled loop
    if value & 0x01: r ^= DUAL_BASIS[0]
    if value & 0x02: r ^= DUAL_BASIS[1]
    if value & 0x04: r ^= DUAL_BASIS[2]
    if value & 0x08: r ^= DUAL_BASIS[3]
    if value & 0x10: r ^= DUAL_BASIS[4]
    if value & 0x20: r ^= DUAL_BASIS[5]
    if value & 0x40: r ^= DUAL_BASIS[6]
    if value & 0x80: r ^= DUAL_BASIS[7]
    return r
