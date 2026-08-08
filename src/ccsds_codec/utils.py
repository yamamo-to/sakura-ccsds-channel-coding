"""Backwards-compatible shim for :mod:`ccsds_codec.core.bits`.

The implementation now lives in :mod:`ccsds_codec.core.bits`; this module is
kept so that ``from ccsds_codec.utils import ...`` keeps working.
"""

from .core.bits import bits_to_bytes, bits_to_bytes_strict, bytes_to_bits

__all__ = ["bytes_to_bits", "bits_to_bytes", "bits_to_bytes_strict"]
