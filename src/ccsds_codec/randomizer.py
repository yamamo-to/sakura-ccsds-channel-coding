"""Backwards-compatible shim for :mod:`ccsds_codec.core.randomizer`.

The implementation now lives in :mod:`ccsds_codec.core.randomizer`; this
module is kept so that ``from ccsds_codec.randomizer import ...`` keeps
working.  The ``main()`` CLI entry point has moved to
:mod:`ccsds_codec.cli` (use ``python -m ccsds_codec`` instead).
"""

from .core.randomizer import MASK, SEED, TAPS, descramble, scramble

__all__ = ["SEED", "MASK", "TAPS", "scramble", "descramble"]
