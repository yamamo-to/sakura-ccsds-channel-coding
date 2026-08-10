"""Unified command-line interface for the CCSDS codec.

Single entry point: ``python -m ccsds_codec``.  All codec modes read binary
data from stdin and write the result to stdout so they can be chained with
ordinary Unix pipes.

Usage examples::

    python -m ccsds_codec conv-enc [--rate 2/3] < in.bin > out.bin
    python -m ccsds_codec conv-dec [--rate 2/3] < out.bin > rec.bin
    python -m ccsds_codec rs-enc    < in.bin > out.bin
    python -m ccsds_codec rs-dec    < out.bin > rec.bin
    python -m ccsds_codec rs-enc --depth 5 < in.bin > out.bin
    python -m ccsds_codec rs-dec --depth 5 < out.bin > rec.bin
    python -m ccsds_codec turbo-enc [--rate 1/6] < in.bin > out.bin
    python -m ccsds_codec turbo-dec [--rate 1/6] < out.bin > rec.bin
    python -m ccsds_codec rand      < in.bin > scrambled.bin
"""

from __future__ import annotations

import argparse
import sys

from .core.bits import bits_to_bytes, bytes_to_bits
from .core.convolutional import (
    decode_byte_padded,
    encode as conv_encode,
    viterbi_decode as conv_viterbi_decode,
)
from .core.randomizer import scramble as randomizer_scramble
from .core.reed_solomon import decode as rs_decode
from .core.reed_solomon import encode as rs_encode
from .core.turbo import (
    decode as turbo_decode,
    decode_padded_rate16,
    decode_unpunctured as turbo_decode_unpunctured,
    encode as turbo_encode,
)

CONV_RATES = ["1/2", "2/3", "3/4", "5/6", "7/8"]
TURBO_RATES = ["1/2", "1/3", "1/4", "1/6"]


def _read_bytes() -> bytes:
    """Read all of stdin as raw bytes."""
    return sys.stdin.buffer.read()


def _write_bytes(data: bytes) -> None:
    """Write raw bytes to stdout."""
    sys.stdout.buffer.write(data)


def _read_bits() -> list[int]:
    """Read all of stdin as an MSB-first bit list."""
    return bytes_to_bits(_read_bytes())


def _write_bits(bits: list[int]) -> None:
    """Pack *bits* into bytes and write them to stdout."""
    _write_bytes(bits_to_bytes(bits))


def _check_rate(parser: argparse.ArgumentParser, rate: str | None, allowed: list[str]) -> None:
    """Abort with a usage error when *rate* is not in *allowed*."""
    if rate is not None and rate not in allowed:
        parser.error(f"invalid rate {rate!r}; choose from {allowed}")


def _conv(mode: str, rate: str | None, parser: argparse.ArgumentParser) -> None:
    """Handle the conv-enc / conv-dec modes."""
    _check_rate(parser, rate, CONV_RATES)
    rate = rate or "1/2"
    bits = _read_bits()
    if mode == "conv-enc":
        _write_bits(conv_encode(bits, rate=rate))
    elif rate == "1/2":
        _write_bits(conv_viterbi_decode(bits, rate=rate))
    else:
        # Punctured rates: up to 7 trailing padding bits may follow the stream.
        _write_bits(decode_byte_padded(bits, rate))


def _rs(mode: str, depth: int) -> None:
    """Handle the rs-enc / rs-dec modes (bytes in, bytes out)."""
    data = _read_bytes()
    if mode == "rs-enc":
        _write_bytes(rs_encode(data, depth=depth))
        return
    try:
        _write_bytes(rs_decode(data, depth=depth))
    except ValueError as e:
        print(f"RS decode error: {e}", file=sys.stderr)
        sys.exit(1)


def _turbo(mode: str, rate: str | None, parser: argparse.ArgumentParser) -> None:
    """Handle the turbo-enc / turbo-dec modes."""
    _check_rate(parser, rate, TURBO_RATES)
    bits = _read_bits()
    if mode == "turbo-enc":
        _write_bits(turbo_encode(bits, rate=rate))
        return
    if rate == "1/6":
        # Rate-1/6 frames are byte-aligned; trim up to 7 trailing pad bits.
        dec = decode_padded_rate16(bits)
    elif rate == "1/3":
        # Rate-1/3 frames carry 4 trailing padding bits when byte-packed.
        dec = turbo_decode_unpunctured(bits)
    elif rate is not None:
        dec = turbo_decode(bits, rate=rate)
    else:
        try:
            # Default encode rate is 1/3; tolerate byte padding.
            dec = turbo_decode_unpunctured(bits)
        except ValueError:
            dec = turbo_decode(bits)
    _write_bits(dec)


def main() -> None:
    """Parse arguments and dispatch to the requested codec mode."""
    parser = argparse.ArgumentParser(prog="ccsds_codec")
    parser.add_argument(
        "mode",
        choices=["conv-enc", "conv-dec", "rs-enc", "rs-dec", "turbo-enc", "turbo-dec", "rand"],
        help="operation mode",
    )
    parser.add_argument("--rate", default=None, help="code rate (e.g. 7/8, 1/6)")
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="RS interleaving depth (default 1)",
    )
    args = parser.parse_args()

    mode = args.mode
    if mode.startswith("conv"):
        _conv(mode, args.rate, parser)
    elif mode.startswith("rs"):
        _rs(mode, args.depth)
    elif mode.startswith("turbo"):
        _turbo(mode, args.rate, parser)
    elif mode == "rand":
        _write_bits(randomizer_scramble(_read_bits()))


if __name__ == "__main__":
    main()
