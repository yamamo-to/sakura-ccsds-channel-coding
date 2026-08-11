"""High-level class-based API for the CCSDS codec library.

Wraps the functional core (:mod:`ccsds_codec.core`) in small, configurable
classes.  Each codec is configured with a frozen dataclass (see
:mod:`ccsds_codec.config`), keeping the settings explicit and typed:

* :class:`RSCodec` – Reed‑Solomon (255,223), configured by :class:`RSConfig`.
* :class:`ConvCodec` – convolutional coding, configured by :class:`ConvConfig`.
* :class:`TurboCodec` – Turbo coding, configured by :class:`TurboConfig`.
* :class:`Randomizer` – CCSDS scrambler, stateless.

The low-level functions remain available from the codec modules
(``ccsds_codec.conv``, ``ccsds_codec.rs``, ...) for pipe-style usage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .config import ConvConfig, RSConfig, TurboConfig
from .core.convolutional import encode as _conv_encode
from .core.convolutional import viterbi_decode as _viterbi_decode
from .core.randomizer import descramble as _descramble
from .core.randomizer import scramble as _scramble
from .core.reed_solomon import decode as _rs_decode
from .core.reed_solomon import encode as _rs_encode
from .core.turbo import decode as _turbo_decode
from .core.turbo import decode_unpunctured as _turbo_decode_unpunctured
from .core.turbo import encode as _turbo_encode

__all__ = [
    "BaseEncoder",
    "BaseDecoder",
    "RSCodec",
    "ConvCodec",
    "TurboCodec",
    "Randomizer",
]

T_in = TypeVar("T_in")
T_out = TypeVar("T_out")


class BaseEncoder(ABC, Generic[T_in, T_out]):
    """Abstract base class for CCSDS encoders.

    All encoder implementations in this package inherit from this class and
    provide a concrete :meth:`encode` method.  The class is generic over the
    codec's input type ``T_in`` and output type ``T_out``.
    """

    @abstractmethod
    def encode(self, data: T_in) -> T_out:
        """Encode *data* and return the encoded representation."""


class BaseDecoder(ABC, Generic[T_in, T_out]):
    """Abstract base class for CCSDS decoders.

    All decoder implementations in this package inherit from this class and
    provide a concrete :meth:`decode` method.  The class is generic over the
    codec's input type ``T_in`` and output type ``T_out``.
    """

    @abstractmethod
    def decode(self, encoded: T_in) -> T_out:
        """Decode *encoded* and return the original representation."""


class RSCodec(BaseEncoder[bytes, bytes], BaseDecoder[bytes, bytes]):
    """Reed‑Solomon (255,223) codec wrapper.

    Applies block interleaving of depth 1…5 as specified by CCSDS 131.0‑B‑4
    §4.3.5 (Figure 4‑2).  A depth of 1 preserves the original behaviour.

    Args:
        config: Codec settings (interleaving depth). Defaults to ``RSConfig()``, i.e. depth 1.
    """

    def __init__(self, config: RSConfig | None = None) -> None:
        self.config = config or RSConfig()

    def encode(self, data: bytes) -> bytes:
        """Encode *data* at the configured interleaving depth."""
        return _rs_encode(data, depth=self.config.depth)

    def decode(self, encoded: bytes) -> bytes:
        """Decode *encoded* at the configured interleaving depth."""
        return _rs_decode(encoded, depth=self.config.depth)


class ConvCodec(BaseEncoder[list[int], list[int]], BaseDecoder[list[int], list[int]]):
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


class TurboCodec(BaseEncoder[list[int], list[int]], BaseDecoder[list[int], list[int]]):
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
