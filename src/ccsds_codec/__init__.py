"""Top‑level package for ccsds_codec.

Exports the most commonly used utility functions so that external code can
import them directly from ``ccsds_codec`` and achieve compatibility with other
open‑source projects that expect these names at the package root.
"""

from .api import ConvCodec, Randomizer, RSCodec, TurboCodec
from .utils import bits_to_bytes, bits_to_bytes_strict, bytes_to_bits

__all__ = [
    "ConvCodec",
    "RSCodec",
    "Randomizer",
    "TurboCodec",
    "bits_to_bytes",
    "bits_to_bytes_strict",
    "bytes_to_bits",
]
