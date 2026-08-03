"""Tests for convolutional encoder/decoder"""

import random

import pytest

from ccsds_codec.conv import G0, G1, encode, viterbi_decode


def test_generator_constants():
    # lsb-current representation: CCSDS G1 = 171_8 (0x79), G2 = 133_8 (0x5B)
    assert G0 == 0x4F
    assert G1 == 0x6D


@pytest.mark.parametrize("length", [1, 5, 10, 20])
def test_encode_decode_roundtrip(length):
    bits = [random.randint(0, 1) for _ in range(length)]
    encoded = encode(bits)
    # Viterbi expects hard bits (0/1) list
    decoded = viterbi_decode(encoded)
    assert decoded == bits
