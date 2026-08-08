"""GF(2^8) arithmetic for the CCSDS Reed‑Solomon code.

Implements a standard GF(2^8) field using the conventional primitive
polynomial ``0x11d`` (the same defaults as the ``reedsolo`` package).
"""

from __future__ import annotations

from typing import List

# Primitive polynomial for the field (same as reedsolo defaults).
PRIMITIVE_POLY = 0x11D  # x^8 + x^4 + x^3 + x^2 + 1
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
