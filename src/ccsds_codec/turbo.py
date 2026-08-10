"""Backwards-compatible shim for :mod:`ccsds_codec.core.turbo`.

The implementation now lives in :mod:`ccsds_codec.core.turbo` (with the CCSDS
§6.3g interleaver in :mod:`ccsds_codec.core.interleaver`); this module is kept
so that ``from ccsds_codec.turbo import ...`` keeps working.  The
``main_encode``/``main_decode`` CLI entry points have moved to
:mod:`ccsds_codec.cli` (use ``python -m ccsds_codec`` instead).
"""

from .core.interleaver import ccsds_deinterleaver, ccsds_interleaver, ccsds_perm
from .core.turbo import (
    GEN,
    GEN2,
    GEN3,
    GEN_SYS,
    LLR_0,
    LLR_1,
    NCOMP,
    STANDARD_K,
    TAIL,
    _bcjr_kernel,  # noqa: F401  (intentional re-export)
    _build_trellis,  # noqa: F401  (intentional re-export)
    _demux,  # noqa: F401  (intentional re-export)
    _detect_rate_k,  # noqa: F401  (intentional re-export)
    _k_for_rate,  # noqa: F401  (intentional re-export)
    _llr_array,  # noqa: F401  (intentional re-export)
    _rsc_streams,  # noqa: F401  (intentional re-export)
    _turbo_decode_core,  # noqa: F401  (intentional re-export)
    decode,
    decode_padded_rate16,
    decode_unpunctured,
    encode,
)

__all__ = [
    "GEN",
    "GEN_SYS",
    "GEN2",
    "GEN3",
    "NCOMP",
    "STANDARD_K",
    "TAIL",
    "LLR_0",
    "LLR_1",
    "ccsds_perm",
    "ccsds_interleaver",
    "ccsds_deinterleaver",
    "encode",
    "decode",
    "decode_unpunctured",
    "decode_padded_rate16",
]
