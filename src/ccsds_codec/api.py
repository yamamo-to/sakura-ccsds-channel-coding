"""High‑level public API for the CCSDS codec library.

Provides thin wrapper classes around the core modules:
- RSCodec for Reed‑Solomon
- ConvCodec for convolutional coding
- TurboCodec for Turbo coding
- Randomizer for the CCSDS scrambler
"""

from __future__ import annotations

from .conv import decode as conv_decode
from .conv import encode as conv_encode  # type: ignore
from .randomizer import descramble as randomizer_descramble
from .randomizer import scramble as randomizer_scramble
from .rs import decode as rs_decode
from .rs import encode as rs_encode
from .turbo import decode as turbo_decode
from .turbo import decode_unpunctured
from .turbo import encode as turbo_encode

__all__ = ["ConvCodec", "RSCodec", "Randomizer", "TurboCodec"]


class RSCodec:
    """Reed‑Solomon (255,223) codec wrapper."""

    @staticmethod
    def encode(data: bytes) -> bytes:
        return rs_encode(data)

    @staticmethod
    def decode(encoded: bytes) -> bytes:
        return rs_decode(encoded)


class ConvCodec:
    """Rate‑1/2 convolutional encoder/decoder wrapper."""

    @staticmethod
    def encode(bits: list[int]) -> list[int]:
        return conv_encode(bits)

    @staticmethod
    def decode(soft_bits: list[int]) -> list[int]:
        return conv_decode(soft_bits)


class TurboCodec:
    """Turbo encoder/decoder wrapper (CCSDS compatible)."""

    @staticmethod
    def encode(bits: list[int], puncture: bool = False) -> list[int]:
        return turbo_encode(bits, puncture=puncture)

    @staticmethod
    def decode(punctured_bits: list[int], iterations: int = 5) -> list[int]:
        return turbo_decode(punctured_bits, iterations=iterations)

    @staticmethod
    def decode_unpunctured(turbo_bits: list[int]) -> list[int]:
        return decode_unpunctured(turbo_bits)


class Randomizer:
    """CCSDS bit randomizer wrapper (scramble/descramble)."""

    @staticmethod
    def scramble(bits: list[int]) -> list[int]:
        return randomizer_scramble(bits)

    @staticmethod
    def descramble(bits: list[int]) -> list[int]:
        return randomizer_descramble(bits)
