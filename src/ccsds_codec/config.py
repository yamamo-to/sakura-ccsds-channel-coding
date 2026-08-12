"""Domain models for the CCSDS codec: code rates and codec configurations.

These value objects make the public API self-documenting: instead of passing
raw strings like ``"7/8"``, callers use :class:`ConvRate` / :class:`TurboRate`
enum members (which are ``str`` subclasses, so they can still be passed to the
legacy functional API), and bundle options in :class:`ConvConfig` /
:class:`TurboConfig` dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = ["ConvRate", "TurboRate", "ConvConfig", "TurboConfig", "RSConfig"]


class ConvRate(StrEnum):
    """Supported convolutional code rates (CCSDS 131.0-B-4)."""

    R1_2 = "1/2"
    R2_3 = "2/3"
    R3_4 = "3/4"
    R5_6 = "5/6"
    R7_8 = "7/8"


class TurboRate(StrEnum):
    """Supported Turbo code rates (CCSDS 131.0-B-4 §3)."""

    R1_2 = "1/2"
    R1_3 = "1/3"
    R1_4 = "1/4"
    R1_6 = "1/6"


@dataclass(frozen=True)
class ConvConfig:
    """Configuration for the convolutional codec.

    Attributes:
        rate: Code rate (default ``1/2``).
        terminate: Whether to append the ``K-1`` zero-flush tail bits.
    """

    rate: ConvRate = ConvRate.R1_2
    terminate: bool = False


@dataclass(frozen=True)
class TurboConfig:
    """Configuration for the Turbo codec.

    Attributes:
        rate: Code rate (default ``1/3``, the full unpunctured scheme).
        iterations: Number of Log-MAP turbo iterations (default 5, CCSDS §3.4 max 10).
    """

    rate: TurboRate = TurboRate.R1_3
    iterations: int = 5

    def __post_init__(self) -> None:
        if not 1 <= self.iterations <= 10:
            raise ValueError(f"iterations must be in 1..10, got {self.iterations}")


@dataclass(frozen=True)
class RSConfig:
    """Configuration for Reed‑Solomon interleaving.

    Attributes:
        depth: Interleaving depth (default 1).
    """

    depth: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.depth <= 5:
            raise ValueError(f"depth must be in 1..5, got {self.depth}")
