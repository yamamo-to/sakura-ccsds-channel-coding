"""Extended tests for convolutional encoder/decoder"""

import random
import pytest
from ccsds_codec.conv import encode, viterbi_decode, G0, G1

def test_encode_length():
    bits = [0, 1, 1, 0, 1]
    enc = encode(bits)
    # Rate 1/2: output length = 2 * input length
    assert len(enc) == 2 * len(bits)

@pytest.mark.parametrize("length", [1, 2, 5, 10, 20, 30])
def test_encode_decode_roundtrip(length):
    bits = [random.randint(0, 1) for _ in range(length)]
    enc = encode(bits)
    dec = viterbi_decode(enc)
    assert dec == bits

def test_generator_constants():
    assert G0 == 0o121
    assert G1 == 0o133
