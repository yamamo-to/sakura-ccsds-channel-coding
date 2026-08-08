"""Top‑level package for ccsds_codec.

Exports the most commonly used names so that external code can import them
directly from ``ccsds_codec``.  The implementation is layered as:

* :mod:`ccsds_codec.core` – pure algorithm modules (bits, galois, interleaver,
  convolutional, reed_solomon, turbo, randomizer).
* :mod:`ccsds_codec.api` – high-level, configurable codec classes.
* :mod:`ccsds_codec.cli` – unified ``python -m ccsds_codec`` command line.
* ``ccsds_codec.conv`` / ``rs`` / ``turbo`` / ``randomizer`` / ``utils`` –
  backwards-compatible shims re-exporting the core modules.
"""

from .api import ConvCodec, RSCodec, Randomizer, TurboCodec
from .config import ConvConfig, ConvRate, TurboConfig, TurboRate
from .core.bits import bits_to_bytes, bits_to_bytes_strict, bytes_to_bits

__all__ = [
    "ConvCodec",
    "RSCodec",
    "Randomizer",
    "TurboCodec",
    "ConvConfig",
    "ConvRate",
    "TurboConfig",
    "TurboRate",
    "bits_to_bytes",
    "bits_to_bytes_strict",
    "bytes_to_bits",
]
