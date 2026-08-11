"""Golden-vector tests for the CCSDS Turbo encoder and interleaver.

Interleaver reference
---------------------
The reference permutation for K = 1784 is the file ``ccsdsSize1784.txt`` from
the ``mdmoctezuma/CCSDSTurboCode`` repository (MATLAB CCSDS 131.0-B-2 Turbo
simulator, https://github.com/mdmoctezuma/CCSDSTurboCode), which contains the
interleaver specified by the CCSDS standard for frame size 1784 bits.  Each
line holds a 1-based input index ``p`` feeding output position ``i``, i.e.
``interleaved[i] = bits[p - 1]``.

The file is committed at ``tests/data/ccsdsSize1784.txt``
(sha256 c7094e3757a60e64c40dc0c0de499315007648be385521bd7a75bf34ae9739fe).

Encoder golden vectors
----------------------
Encoder output vectors are generated from the independent C reference
implementation ``geeanlooca/deepspace-turbo``
(https://github.com/geeanlooca/deepspace-turbo), which implements CCSDS
131.0-B-2 Turbo codes.  The deterministic input used for the vectors is the
alternating bit sequence ``[0, 1, 0, 1, ...]``.

Rates 1/3, 1/4 and 1/6 are taken directly from that reference.  Rate 1/2 is
derived by applying the CCSDS 131.0-B-4 §3.4 puncturing pattern
``(out 0a, out 1a, out 0a, out 1b)`` to the rate-1/3 reference vectors,
because the puncturing pattern used by ``deepspace-turbo`` for rate 1/2
differs from the CCSDS standard.

Decoder reference vectors
-------------------------
``deepspace-turbo`` also implements the turbo decoder (``turbo_decode``,
iterative Log-MAP/BCJR over the same RSC constituents and interleaver).
The reference decoder output vectors below were produced with the
``scripts/turbo_decoder_reference/decode_driver.c`` driver against that
implementation (K = 1784, 3568, 7136 and 8920, 10 iterations, noise
variance 0.25, BPSK symbols with bit 0 -> -1, bit 1 -> +1):

* ``turbo_k{K}_r{rcode}_err{nerr}_rx.txt`` – the received hard bit stream
  (0/1, ``NCOMP[rate] * (K + 4)`` bits) built from the encoder golden
  payload ``[i % 2 for i in range(K)]`` (payload identical to the encoder
  golden vectors) with a deterministic number of bit errors injected at
  fixed positions (8 errors: ``numpy.random.default_rng(7)``; 30 errors:
  ``numpy.random.default_rng(11)``);
* ``turbo_k{K}_r{rcode}_err{nerr}_dec.txt`` – the payload bits recovered by
  the reference decoder for that received stream (``K`` bits).

K is the set of CCSDS 131.0-B-4 Table 6-1 standard information block
lengths 1784/3568/7136/8920.  K = 16384 is deliberately *not* covered:
it is not a standard Turbo information block length in CCSDS 131.0-B-4
(it belongs to the LDPC family, Table 7-1), the reference implementation
rejects it (K % 1784 != 0), and no independent decoder for it is known.

The tests below assert that :func:`ccsds_codec.core.turbo.decode` recovers
exactly those reference bits (with the default 5 iterations), i.e. both
implementations agree bit-for-bit on error-corrected frames for every CCSDS
rate whose puncturing is standard (1/3, 1/4, 1/6; rate 1/2 is excluded
because the reference uses a non-standard puncturing pattern).
"""

from pathlib import Path

import pytest

from ccsds_codec.core.interleaver import ccsds_perm
from ccsds_codec.core.turbo import STANDARD_K, decode, encode

DATA_DIR = Path(__file__).parent / "data"
REF_K1784 = DATA_DIR / "ccsdsSize1784.txt"


def _load_ref(path: Path) -> list[int]:
    """Load a 1-based reference permutation as 0-based indices."""
    ref = [int(line.strip()) for line in path.read_text().splitlines()]
    return [p - 1 for p in ref]


def _load_bitstring(path: Path) -> str:
    """Load a reference bit string from a text file."""
    return path.read_text().strip()


def test_ccsds_perm_1784_matches_reference():
    """K=1784 permutation is bit-exact against the CCSDS reference table."""
    ref = _load_ref(REF_K1784)
    assert len(ref) == 1784
    ours = ccsds_perm(1784)
    assert ours == ref


def test_ccsds_perm_1784_is_bijective():
    """The golden permutation is a true bijection (all 1784 indices)."""
    ours = ccsds_perm(1784)
    assert sorted(ours) == list(range(1784))


@pytest.mark.parametrize("K", STANDARD_K)
@pytest.mark.parametrize("rate", ["1/2", "1/3", "1/4", "1/6"])
def test_turbo_encode_matches_golden_vector(K: int, rate: str) -> None:
    """Encoder output matches the reference vector for the selected rate."""
    rcode = rate.replace("/", "")
    golden = _load_bitstring(DATA_DIR / f"turbo_k{K}_r{rcode}_golden.txt")
    bits = [i % 2 for i in range(K)]
    ours = encode(bits, rate=rate)
    assert len(ours) == len(golden)
    assert "".join(str(b) for b in ours) == golden


DECODER_K = (1784, 3568, 7136, 8920)


@pytest.mark.parametrize("K", DECODER_K)
@pytest.mark.parametrize("rate", ["1/3", "1/4", "1/6"])
@pytest.mark.parametrize("nerr", [8, 30])
def test_turbo_decode_matches_reference_decoder(K: int, rate: str, nerr: int) -> None:
    """Decoder output matches the deepspace-turbo reference decoder.

    The received stream is the encoder golden payload ``[i % 2 ...]``
    encoded at the given rate with ``nerr`` deterministic bit errors.  The
    reference decoder recovers the payload exactly, and our decoder must
    produce the same bits (bit-for-bit agreement on the corrected frame).
    """
    rcode = rate.replace("/", "")
    rx = _load_bitstring(DATA_DIR / f"turbo_k{K}_r{rcode}_err{nerr}_rx.txt")
    ref = _load_bitstring(DATA_DIR / f"turbo_k{K}_r{rcode}_err{nerr}_dec.txt")
    ours = decode([int(c) for c in rx], rate=rate)
    assert "".join(str(b) for b in ours) == ref
