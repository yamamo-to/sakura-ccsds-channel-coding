"""Known‑vector tests for the CCSDS convolutional encoder (Rate 1/2, K=7).

The expected bit streams were generated with the C++ ``ViterbiCodec`` from the
gr-satellites repository (polys ``[0x4F, 0x6D]``) followed by inversion of the
second output symbol, i.e. the on‑air convention that gr-satellites decodes
with GNU Radio ``fec.cc_decoder`` (polys ``[79, -109]``). See AGENTS.md §B.
"""

import pytest

from ccsds_codec.conv import G0, G1, encode, encode_cxx

# Expected channel streams (terminated, 2 * (len + K - 1) symbols)
TERMINATED_VECTORS = {
    "ALT_10101010": (
        [1, 0, 1, 0, 1, 0, 1, 0],
        [
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
        ],
    ),
    "SHORT_10110": (
        [1, 0, 1, 1, 0],
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
    ),
    "SINGLE_1": (
        [1],
        [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    ),
    "ALLZERO_4": (
        [0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ),
    "ALLONE_8": (
        [1, 1, 1, 1, 1, 1, 1, 1],
        [
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
        ],
    ),
}


@pytest.mark.parametrize("case", sorted(TERMINATED_VECTORS))
def test_encode_terminated_known_vector(case):
    bits, expected = TERMINATED_VECTORS[case]
    assert encode(bits, terminate=True) == expected
    assert encode_cxx(bits, terminate=True) == expected


@pytest.mark.parametrize("case", sorted(TERMINATED_VECTORS))
def test_encode_unterminated_known_vector(case):
    bits, expected = TERMINATED_VECTORS[case]
    # non-terminated stream = first 2 * len(bits) symbols of the terminated one
    assert encode(bits) == expected[: 2 * len(bits)]


def test_generator_constants():
    assert G0 == 0x4F  # lsb-current CCSDS G1 = 171_8
    assert G1 == 0x6D  # lsb-current CCSDS G2 = 133_8 (inverted on the channel)
