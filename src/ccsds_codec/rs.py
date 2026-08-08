"""Backwards-compatible shim for :mod:`ccsds_codec.core.reed_solomon`.

The implementation now lives in :mod:`ccsds_codec.core.reed_solomon`; this
module is kept so that ``from ccsds_codec.rs import ...`` keeps working.
The ``main_encode``/``main_decode`` CLI entry points have moved to
:mod:`ccsds_codec.cli` (use ``python -m ccsds_codec`` instead).
"""

from .core.galois import (
    EXP_TABLE,
    GF_SIZE,
    LOG_TABLE,
    PRIMITIVE_POLY,
    gf_add,
    gf_inverse,
    gf_mul,
    gf_pow,
    gf_sub,
)
from .core.reed_solomon import (
    GENERATOR,
    RS_K,
    RS_N,
    RS_SYMS,
    decode,
    decode_block,
    encode,
    encode_block,
)
from .core.reed_solomon import decode_block as _rs_decode_block  # noqa: F401
from .core.reed_solomon import encode_block as _rs_encode_block  # noqa: F401

__all__ = [
    "RS_N",
    "RS_K",
    "RS_SYMS",
    "GENERATOR",
    "PRIMITIVE_POLY",
    "GF_SIZE",
    "EXP_TABLE",
    "LOG_TABLE",
    "gf_add",
    "gf_sub",
    "gf_mul",
    "gf_pow",
    "gf_inverse",
    "encode",
    "decode",
    "encode_block",
    "decode_block",
]
