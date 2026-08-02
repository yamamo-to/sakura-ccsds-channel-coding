"""Top‑level package for ccsds_codec.

Exports the most commonly used utility functions so that external code can
import them directly from ``ccsds_codec`` and achieve compatibility with other
open‑source projects that expect these names at the package root.
"""

from .utils import bytes_to_bits, bits_to_bytes, bits_to_bytes_strict
from .api import RSCodec, ConvCodec, TurboCodec, Randomizer

__all__ = [
    "bytes_to_bits",
    "bits_to_bytes",
    "bits_to_bytes_strict",
    "RSCodec",
    "ConvCodec",
    "TurboCodec",
    "Randomizer",
]
