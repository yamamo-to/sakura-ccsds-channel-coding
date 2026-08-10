"""Golden-vector tests for the CCSDS Turbo interleaver (CCSDS 131.0-B-4 §6.3g).

The reference permutation for K = 1784 is the file ``ccsdsSize1784.txt`` from
the ``mdmoctezuma/CCSDSTurboCode`` repository (MATLAB CCSDS 131.0-B-2 Turbo
simulator, https://github.com/mdmoctezuma/CCSDSTurboCode), which contains the
interleaver specified by the CCSDS standard for frame size 1784 bits.  Each
line holds a 1-based input index ``p`` feeding output position ``i``, i.e.
``interleaved[i] = bits[p - 1]``.

The file is committed at ``tests/data/ccsdsSize1784.txt``
(sha256 c7094e3757a60e64c40dc0c0de499315007648be385521bd7a75bf34ae9739fe).
"""

from pathlib import Path

from ccsds_codec.core.interleaver import ccsds_perm

DATA_DIR = Path(__file__).parent / "data"
REF_K1784 = DATA_DIR / "ccsdsSize1784.txt"


def _load_ref(path: Path) -> list[int]:
    """Load a 1-based reference permutation as 0-based indices."""
    ref = [int(line.strip()) for line in path.read_text().splitlines()]
    return [p - 1 for p in ref]


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
