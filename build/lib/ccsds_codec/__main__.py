"""CLI entry point for the ccsds_codec package.

Usage examples:

    python -m ccsds_codec turbo-enc < input.bin > out.turbo
    python -m ccsds_codec turbo-dec < out.turbo > recovered.bin
    python -m ccsds_codec conv-enc  < input.bin > out.conv
    python -m ccsds_codec conv-dec  < out.conv  > recovered.bin
    python -m ccsds_codec rs-enc    < input.bin > out.rs
    python -m ccsds_codec rs-dec    < out.rs    > recovered.bin
    python -m ccsds_codec rand      < input.bin > scrambled.bin
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

def main() -> None:
    parser = argparse.ArgumentParser(prog="ccsds_codec")
    parser.add_argument("mode", choices=sorted(MAP.keys()), help="operation mode")
    args = parser.parse_args()
    MAP[args.mode]()

if __name__ == "__main__":
    main()
