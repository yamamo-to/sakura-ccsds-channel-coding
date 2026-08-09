"""Basic tests for convolutional encoder and Viterbi decoder."""

import random
import pytest

from ccsds_codec import conv


def test_parity_known():
    # parity of 0 is 0
    assert conv._parity(0) == 0
    # parity of 0b1011 (11) is 1 (odd number of ones)
    assert conv._parity(0b1011) == 1
    # parity of 0b1110 (14) is 1 (three ones)
    assert conv._parity(0b1110) == 1


def test_encode_decode_roundtrip():
    # generate a random bit sequence of length multiple of 8
    random.seed(0)
    bits = [random.randint(0, 1) for _ in range(32)]
    encoded = conv.encode(bits)
    # Viterbi decoder expects hard bits (0/1) list
    decoded = conv.viterbi_decode(encoded)
    assert decoded == bits


def test_viterbi_invalid_length():
    with pytest.raises(ValueError):
        conv.viterbi_decode([0, 1, 0])  # odd length
