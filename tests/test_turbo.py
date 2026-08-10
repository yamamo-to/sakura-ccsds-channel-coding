"""Tests for Turbo encoder/decoder"""

import random

import pytest

from ccsds_codec.turbo import decode, decode_unpunctured, encode


@pytest.mark.parametrize("length", [8, 16, 24])
def test_unpunctured_roundtrip(length):
    bits = [random.randint(0, 1) for _ in range(length)]
    turbo_bits = encode(bits, puncture=False)  # rate 1/3, 3*(K+4) bits
    assert len(turbo_bits) == 3 * (length + 4)
    # decode_unpunctured returns payload bits (hard decision)
    decoded = decode_unpunctured(turbo_bits)
    assert decoded == bits


@pytest.mark.parametrize("length", [8, 16, 24])
def test_punctured_roundtrip(length):
    bits = [random.randint(0, 1) for _ in range(length)]
    punctured = encode(bits, puncture=True)  # rate 1/2, 2*(K+4) bits
    assert len(punctured) == 2 * (length + 4)
    # non-standard K requires an explicit rate
    decoded = decode(punctured, iterations=3, rate="1/2")
    assert decoded == bits
