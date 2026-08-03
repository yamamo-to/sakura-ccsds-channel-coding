"""Utility functions for bit/byte conversions used across the CCSDS codec modules.

The functions operate on Python ``bytes`` objects and ``list[int]`` bit lists where each bit is ``0`` or ``1``.
"""

from __future__ import annotations


def bytes_to_bits(data: bytes) -> list[int]:
    """Convert a ``bytes`` object to a list of bits (MSB first for each byte)."""
    bits: list[int] = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def bits_to_bytes(bits: list[int]) -> bytes:
    """Pack a list of bits (MSB first) into a ``bytes`` object.

    The input list is **not** mutated. If its length is not a multiple of 8 it
    is padded with zeros (least‑significant bits) to the next byte boundary.
    """
    bits = list(bits)
    if any(b not in (0, 1) for b in bits):
        raise ValueError("bits list must contain only 0 or 1 values")
    if len(bits) % 8 != 0:
        pad_len = 8 - (len(bits) % 8)
        bits.extend([0] * pad_len)
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i : i + 8]:
            byte = (byte << 1) | bit
        out.append(byte)
    return bytes(out)


def bits_to_bytes_strict(bits: list[int]) -> bytes:
    """Strict version of :func:`bits_to_bytes`.

    Validates the input and **does not** pad; the length must be a multiple of 8.
    Raises :class:`ValueError` otherwise. This mirrors the behaviour of libraries
    that expect full-byte aligned bit streams.
    """
    bits = list(bits)
    if any(b not in (0, 1) for b in bits):
        raise ValueError("bits list must contain only 0 or 1 values")
    if len(bits) % 8 != 0:
        raise ValueError("bits length must be a multiple of 8 for strict conversion")
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i : i + 8]:
            byte = (byte << 1) | bit
        out.append(byte)
    return bytes(out)
