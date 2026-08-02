"""Tests for convolutional encoder/decoder"""

import random
import pytest
from ccsds_codec.conv import encode, viterbi_decode, G0, G1

def test_generator_constants():
    assert G0 == 0o121  # 81 decimal, per CCSDS spec
    assert G1 == 0o133  # 91 decimal

@pytest.mark.parametrize("length", [1, 5, 10, 20])
def test_encode_decode_roundtrip(length):
    bits = [random.randint(0, 1) for _ in range(length)]
    encoded = encode(bits)
    # Viterbi expects hard bits (0/1) list
    decoded = viterbi_decode(encoded)
    assert decoded == bits
