#!/usr/bin/env python3
"""Regenerate the Turbo decoder reference golden vectors.

The vectors ``tests/data/turbo_k{K}_r{rcode}_err{nerr}_{rx,dec}.txt`` are
produced by the independent C reference implementation
``geeanlooca/deepspace-turbo`` (CCSDS 131.0-B-2 Turbo codes), driven through
the ``scripts/turbo_decoder_reference/decode_driver.c`` wrapper.  This script
reproduces them deterministically:

1. payload  = alternating bits ``[i % 2 for i in range(K)]`` (the same input
   used for the encoder golden vectors);
2. codeword = this library's ``encode(payload, rate=...)`` (verified to be
   bit-identical to the reference encoder);
3. received = codeword with ``nerr`` bit errors injected at fixed positions
   drawn from a seeded ``numpy.random.default_rng`` (8 errors: seed 7,
   30 errors: seed 11);
4. reference decode = ``decode_driver decode K rate 10 0.25`` on the BPSK
   symbols of *received* (bit 0 -> -1, bit 1 -> +1), which runs the
   reference ``turbo_decode`` with 10 iterations and noise variance 0.25.

Usage:
    python scripts/gen_turbo_decoder_golden.py <deepspace-turbo-dir>

``<deepspace-turbo-dir>`` must be a checkout of
https://github.com/geeanlooca/deepspace-turbo containing ``libturbocodes.c``,
``libconvcodes.c`` and ``utilities.c``.  Requires a C compiler (``gcc``).

Block lengths are the CCSDS 131.0-B-4 Table 6-1 standard lengths
1784/3568/7136/8920.  K = 16384 is deliberately excluded: it is *not* a
standard Turbo information block length (it belongs to the LDPC family,
Table 7-1) and the reference implementation rejects it (K % 1784 != 0).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from ccsds_codec.core.turbo import encode

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "tests" / "data"
DRIVER_SRC = ROOT / "scripts" / "turbo_decoder_reference" / "decode_driver.c"

KS = (1784, 3568, 7136, 8920)
RATES = ["1/3", "1/4", "1/6"]
ERROR_CONFIGS = [(8, 7), (30, 11)]


def build_driver(ref_dir: Path, workdir: Path) -> Path:
    """Compile decode_driver.c against the deepspace-turbo sources."""
    sources = " ".join(
        str(ref_dir / f) for f in ("libturbocodes.c", "libconvcodes.c", "utilities.c")
    )
    out = workdir / "decode_driver"
    subprocess.run(
        f"gcc -O2 -I{ref_dir} -o {out} {DRIVER_SRC} {sources} -lm",
        shell=True,
        check=True,
        capture_output=True,
    )
    return out


def reference_decode(driver: Path, K: int, rate: str, received: np.ndarray, workdir: Path) -> str:
    """Run the reference decoder and return the recovered payload bit string."""
    sym_path = workdir / f"sym_K{K}_{rate.replace('/', '_')}.txt"
    out_path = workdir / f"dec_K{K}_{rate.replace('/', '_')}.txt"
    sym = 2.0 * received.astype(np.float64) - 1.0
    sym_path.write_text(" ".join(f"{v:.1f}" for v in sym) + "\n")
    subprocess.run(
        [str(driver), "decode", str(K), rate, "10", "0.25", str(sym_path), str(out_path)],
        check=True,
        capture_output=True,
    )
    bits = out_path.read_text().strip().replace(" ", "")
    if len(bits) != K:
        raise RuntimeError(f"reference decoder returned {len(bits)} bits, expected {K}")
    return bits


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    ref_dir = Path(sys.argv[1]).resolve()
    for f in ("libturbocodes.c", "libconvcodes.c", "utilities.c"):
        if not (ref_dir / f).exists():
            print(f"error: {ref_dir / f} not found (need a deepspace-turbo checkout)")
            return 2

    workdir = ROOT / "build" / "turbo_decoder_golden"
    workdir.mkdir(parents=True, exist_ok=True)
    driver = build_driver(ref_dir, workdir)

    for K in KS:
        payload = np.array([i % 2 for i in range(K)], dtype=np.uint8)
        for rate in RATES:
            codeword = np.array(encode(payload.tolist(), rate=rate), dtype=np.uint8)
            for nerr, seed in ERROR_CONFIGS:
                rng = np.random.default_rng(seed)
                pos = sorted(rng.choice(len(codeword), size=nerr, replace=False).tolist())
                received = codeword.copy()
                for p in pos:
                    received[p] ^= 1
                ref = reference_decode(driver, K, rate, received, workdir)

                rcode = rate.replace("/", "")
                (DATA_DIR / f"turbo_k{K}_r{rcode}_err{nerr}_rx.txt").write_text(
                    "".join(map(str, received)) + "\n"
                )
                (DATA_DIR / f"turbo_k{K}_r{rcode}_err{nerr}_dec.txt").write_text(ref + "\n")
                print(f"K={K} rate {rate} nerr={nerr}: wrote rx/dec ({len(ref)} payload bits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
