"""High-level class-based API for the CCSDS codec library.

Wraps the functional core (:mod:`ccsds_codec.core`) in small, configurable
classes.  Each codec is configured with a frozen dataclass (see
:mod:`ccsds_codec.config`), keeping the settings explicit and typed:

* :class:`RSCodec` – Reed‑Solomon (255,223), stateless.
* :class:`ConvCodec` – convolutional coding, configured by :class:`ConvConfig`.
* :class:`TurboCodec` – Turbo coding, configured by :class:`TurboConfig`.
* :class:`Randomizer` – CCSDS scrambler, stateless.

The low-level functions remain available from the codec modules
(``ccsds_codec.conv``, ``ccsds_codec.rs``, ...) for pipe-style usage.
"""

from __future__ import annotations

from .config import ConvConfig, TurboConfig
from .core.convolutional import encode as _conv_encode
from .core.convolutional import viterbi_decode as _viterbi_decode
from .core.randomizer import descramble as _descramble
from .core.randomizer import scramble as _scramble
from .core.reed_solomon import decode as _rs_decode
from .core.reed_solomon import encode as _rs_encode
from .core.turbo import decode as _turbo_decode
from .core.turbo import decode_unpunctured as _turbo_decode_unpunctured
from .core.turbo import encode as _turbo_encode

__all__ = ["RSCodec", "ConvCodec", "TurboCodec", "Randomizer"]


class RSCodec:
    """Reed‑Solomon (255,223) codec wrapper.

    Stateless: encoding/decoding operate on whole byte streams and are exposed
    as static methods.
    """

    @staticmethod
    def encode(data: bytes) -> bytes:
        """Encode *data*, splitting it into ``RS_K``-byte blocks."""
        return _rs_encode(data)

    @staticmethod
    def decode(encoded: bytes) -> bytes:
        """Decode *encoded* back into the data portion of each block."""
        return _rs_decode(encoded)


class ConvCodec:
    """Convolutional encoder/decoder (rates 1/2, 2/3, 3/4, 5/6, 7/8).

    Args:
        config: Codec settings (rate, termination).  Defaults to
            ``ConvConfig()``, i.e. rate ``1/2`` without tail.
    """

    def __init__(self, config: ConvConfig | None = None) -> None:
        self.config = config or ConvConfig()

    def encode(self, bits: list[int]) -> list[int]:
        """Encode *bits* at the configured rate."""
        return _conv_encode(bits, terminate=self.config.terminate, rate=self.config.rate)

    def decode(self, soft_bits: list[int]) -> list[int]:
        """Viterbi-decode *soft_bits* at the configured rate."""
        return _viterbi_decode(soft_bits, rate=self.config.rate)


class TurboCodec:
    """Turbo encoder/decoder (rates 1/2, 1/3, 1/4, 1/6).

    Args:
        config: Codec settings (rate, iteration count).  Defaults to
            ``TurboConfig()``, i.e. full rate ``1/3`` with 5 iterations.
    """

    def __init__(self, config: TurboConfig | None = None) -> None:
        self.config = config or TurboConfig()

    def encode(self, bits: list[int]) -> list[int]:
        """Encode *bits* at the configured rate."""
        return _turbo_encode(bits, rate=self.config.rate)

    def decode(self, punctured_bits: list[int]) -> list[int]:
        """Decode *punctured_bits* at the configured rate."""
        return _turbo_decode(
            punctured_bits, iterations=self.config.iterations, rate=self.config.rate
        )

    def decode_unpunctured(self, turbo_bits: list[int]) -> list[int]:
        """Decode a full rate‑1/3 stream, ignoring the configured rate."""
        return _turbo_decode_unpunctured(turbo_bits)


class Randomizer:
    """CCSDS bit randomizer wrapper (scramble/descramble).

    Stateless: the LFSR is re-seeded to the all-ones state on every call.
    """

    @staticmethod
    def scramble(bits: list[int]) -> list[int]:
        """Scramble *bits* with the CCSDS pseudo-random sequence."""
        return _scramble(bits)

    @staticmethod
    def descramble(bits: list[int]) -> list[int]:
        """Descramble *bits* (the CCSDS randomizer is its own inverse)."""
        return _descramble(bits)
