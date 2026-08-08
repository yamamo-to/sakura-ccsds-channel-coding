"""Backwards-compatible shim for :mod:`ccsds_codec.core.turbo`.

The implementation now lives in :mod:`ccsds_codec.core.turbo` (with the QPP
interleaver in :mod:`ccsds_codec.core.interleaver`); this module is kept so
that ``from ccsds_codec.turbo import ...`` keeps working.  The
``main_encode``/``main_decode`` CLI entry points have moved to
:mod:`ccsds_codec.cli` (use ``python -m ccsds_codec`` instead).
"""

from .core.interleaver import ccsds_deinterleaver, ccsds_interleaver
from .core.turbo import (
    GEN,
    GEN2,
    GEN3,
    LLR_0,
    LLR_1,
    MASK,
    TAIL,
    _bcjr_kernel,  # noqa: F401  (intentional re-export)
    _bcjr_multi_kernel,  # noqa: F401  (intentional re-export)
    _decode_core,  # noqa: F401  (intentional re-export)
    _decode_core_rate16,  # noqa: F401  (intentional re-export)
    _depuncture,  # noqa: F401  (intentional re-export)
    _llr_array,  # noqa: F401  (intentional re-export)
    _puncture,  # noqa: F401  (intentional re-export)
    _rsc_parity,  # noqa: F401  (intentional re-export)
    _rsc_parities,  # noqa: F401  (intentional re-export)
    decode,
    decode_padded_rate16,
    decode_unpunctured,
    encode,
    payload_len_from_punctured,
)

__all__ = [
    "GEN",
    "MASK",
    "TAIL",
    "GEN2",
    "GEN3",
    "LLR_0",
    "LLR_1",
    "payload_len_from_punctured",
    "ccsds_interleaver",
    "ccsds_deinterleaver",
    "encode",
    "decode",
    "decode_unpunctured",
    "decode_padded_rate16",
]
