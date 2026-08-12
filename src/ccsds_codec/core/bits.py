"""Bit/byte conversion primitives shared across the CCSDS codec modules.

The functions operate on Python ``bytes`` objects and ``list[int]`` bit lists
where each bit is ``0`` or ``1``.  Bit order is MSB-first within each byte.
"""

from __future__ import annotations

__all__ = ["bytes_to_bits", "bits_to_bytes", "bits_to_bytes_strict"]


def bytes_to_bits(data: bytes) -> list[int]:
    """Convert a ``bytes`` object to a list of bits (MSB first per byte)."""
    bits: list[int] = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def _validate_bits(bits: list[int]) -> None:
    """Raise :class:`ValueError` unless every element of *bits* is 0 or 1."""
    for i, b in enumerate(bits):
        if b not in (0, 1):
            raise ValueError(f"Bit at position {i} is not 0 or 1: {b}")


def _pack_bits(bits: list[int]) -> bytes:
    """Pack 8-aligned bits (MSB first) into bytes without validation."""
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i : i + 8]:
            byte = (byte << 1) | bit
        out.append(byte)
    return bytes(out)


def bits_to_bytes(bits: list[int]) -> bytes:
    """Pack a list of bits (MSB first) into a ``bytes`` object.

    The input list is **not** mutated. If its length is not a multiple of 8 it
    is padded with zeros (least‑significant bits) to the next byte boundary.
    """
    bits = list(bits)
    _validate_bits(bits)
    if len(bits) % 8 != 0:
        bits.extend([0] * (8 - (len(bits) % 8)))
    return _pack_bits(bits)


def bits_to_bytes_strict(bits: list[int]) -> bytes:
    """Strict version of :func:`bits_to_bytes`.

    Validates the input and **does not** pad; the length must be a multiple of 8.
    Raises :class:`ValueError` otherwise. This mirrors the behaviour of libraries
    that expect full-byte aligned bit streams.
    """
    _validate_bits(bits)
    if len(bits) % 8 != 0:
        raise ValueError("bits length must be a multiple of 8 for strict conversion")
    return _pack_bits(bits)
