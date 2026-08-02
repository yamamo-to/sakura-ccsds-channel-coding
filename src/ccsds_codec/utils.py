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

    If the number of bits is not a multiple of 8 the last byte is padded with zeros on the right.
    """
    if len(bits) % 8 != 0:
        # Pad with zeros (least‑significant bits) to make a full byte
        pad_len = 8 - (len(bits) % 8)
        bits = bits + [0] * pad_len
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i : i + 8]:
            byte = (byte << 1) | bit
        out.append(byte)
    return bytes(out)
