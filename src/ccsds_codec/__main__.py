"""CLI entry point for the ccsds_codec package.

Usage examples:

    python -m ccsds_codec turbo-enc < input.bin > out.turbo
    python -m ccsds_codec turbo-dec < out.turbo > recovered.bin
    python -m ccsds_codec conv-enc  < input.bin > out.conv
    python -m ccsds_codec conv-dec  < out.conv  > recovered.bin
    python -m ccsds_codec rs-enc    < input.bin > out.rs
    python -m ccsds_codec rs-dec    < out.rs    > recovered.bin
    python -m ccsds_codec rand      < input.bin > scrambled.bin

All conv/turbo modes accept ``--rate`` (e.g. ``--rate 7/8`` for
convolutional punctured rates, ``--rate 1/6`` for the full Turbo code).
"""

import argparse
from .conv import main_encode as conv_enc, main_decode as conv_dec
from .rs import main_encode as rs_enc, main_decode as rs_dec
from .turbo import main_encode as turbo_enc, main_decode as turbo_dec
from .randomizer import main as rand_main

MAP = {
    "conv-enc": conv_enc,
    "conv-dec": conv_dec,
    "rs-enc": rs_enc,
    "rs-dec": rs_dec,
    "turbo-enc": turbo_enc,
    "turbo-dec": turbo_dec,
    "rand": rand_main,
}

CONV_RATES = ["1/2", "2/3", "3/4", "5/6", "7/8"]
TURBO_RATES = ["1/2", "1/3", "1/4", "1/6"]


def main() -> None:
    parser = argparse.ArgumentParser(prog="ccsds_codec")
    parser.add_argument("mode", choices=sorted(MAP.keys()), help="operation mode")
    parser.add_argument(
        "--rate",
        default=None,
        help="code rate for conv/turbo modes (e.g. 7/8, 1/6)",
    )
    args = parser.parse_args()
    mode = args.mode
    rate = args.rate
    if rate is None:
        MAP[mode]()
        return
    if mode.startswith("conv"):
        if rate not in CONV_RATES:
            parser.error(f"invalid conv rate {rate!r}; choose from {CONV_RATES}")
        MAP[mode](rate)
    elif mode.startswith("turbo"):
        if rate not in TURBO_RATES:
            parser.error(f"invalid turbo rate {rate!r}; choose from {TURBO_RATES}")
        MAP[mode](rate)
    else:
        parser.error(f"--rate is not applicable to mode {mode!r}")


if __name__ == "__main__":
    main()
