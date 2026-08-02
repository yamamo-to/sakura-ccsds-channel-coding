"""Tests for Turbo encoder/decoder"""

import random
import pytest
from ccsds_codec.turbo import encode, decode, decode_unpunctured

@pytest.mark.parametrize("length", [5, 10, 20])
def test_unpunctured_roundtrip(length):
    bits = [random.randint(0, 1) for _ in range(length)]
    turbo_bits = encode(bits, puncture=False)
    # decode_unpunctured returns payload bits (hard decision)
    decoded = decode_unpunctured(turbo_bits)
    assert decoded == bits

@pytest.mark.parametrize("length", [5, 10, 20])
def test_punctured_roundtrip(length):
    bits = [random.randint(0, 1) for _ in range(length)]
    punctured = encode(bits, puncture=True)
    decoded = decode(punctured, iterations=3)
    assert decoded == bits
