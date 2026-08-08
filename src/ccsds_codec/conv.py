"""Backwards-compatible shim for :mod:`ccsds_codec.core.convolutional`.

The implementation now lives in :mod:`ccsds_codec.core.convolutional`; this
module is kept so that ``from ccsds_codec.conv import ...`` keeps working.
The ``main_encode``/``main_decode`` CLI entry points have moved to
:mod:`ccsds_codec.cli` (use ``python -m ccsds_codec`` instead).
"""

from .core.convolutional import (
    G0,
    G1,
    K,
    MASK,
    PUNCTURE_PATTERNS,
    _build_tables,  # noqa: F401  (intentional re-export)
    _depuncture,  # noqa: F401  (intentional re-export)
    _parity,  # noqa: F401  (intentional re-export)
    _viterbi_hard_kernel,  # noqa: F401  (intentional re-export)
    _viterbi_llr_kernel,  # noqa: F401  (intentional re-export)
    decode,
    decode_byte_padded,
    encode,
    encode_cxx,
    viterbi_decode,
)

__all__ = [
    "G0",
    "G1",
    "K",
    "MASK",
    "PUNCTURE_PATTERNS",
    "encode",
    "encode_cxx",
    "decode",
    "viterbi_decode",
    "decode_byte_padded",
]
