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
Encoder output vectors for the unpunctured rates 1/3, 1/4 and 1/6 and for all
five CCSDS block lengths are generated from the independent C reference
implementation ``geeanlooca/deepspace-turbo``
(https://github.com/geeanlooca/deepspace-turbo), which implements CCSDS
131.0-B-2 Turbo codes.  The deterministic input used for the vectors is the
alternating bit sequence ``[0, 1, 0, 1, ...]``.  Rate 1/2 is not included
because the puncturing pattern used by that reference differs from the CCSDS
standard pattern; the standard pattern is verified structurally in the encoder
tests instead.
"""

from pathlib import Path

import pytest

from ccsds_codec.core.interleaver import ccsds_perm
from ccsds_codec.core.turbo import STANDARD_K, encode

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
@pytest.mark.parametrize("rate", ["1/3", "1/4", "1/6"])
def test_turbo_encode_matches_golden_vector(K: int, rate: str) -> None:
    """Encoder output matches an independent reference for rates 1/3, 1/4, 1/6."""
    rcode = rate.replace("/", "")
    golden = _load_bitstring(DATA_DIR / f"turbo_k{K}_r{rcode}_golden.txt")
    bits = [i % 2 for i in range(K)]
    ours = encode(bits, rate=rate)
    assert len(ours) == len(golden)
    assert "".join(str(b) for b in ours) == golden
