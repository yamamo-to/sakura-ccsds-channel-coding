"""Tests for punctured convolutional code rates (2/3, 3/4, 5/6, 7/8)."""

import random

import pytest

from ccsds_codec.conv import PUNCTURE_PATTERNS, encode, viterbi_decode
from ccsds_codec.conv import _decode_byte_padded

PUNCTURED_RATES = ["2/3", "3/4", "5/6", "7/8"]


@pytest.mark.parametrize("rate", PUNCTURED_RATES)
@pytest.mark.parametrize("length", [1, 2, 7, 32, 100])
def test_punctured_roundtrip(rate, length):
    bits = [random.randint(0, 1) for _ in range(length)]
    enc = encode(bits, rate=rate)
    dec = viterbi_decode(enc, rate=rate)
    assert dec == bits


@pytest.mark.parametrize("rate", PUNCTURED_RATES)
def test_punctured_roundtrip_terminated(rate):
    bits = [random.randint(0, 1) for _ in range(64)]
    enc = encode(bits, terminate=True, rate=rate)
    dec = viterbi_decode(enc, rate=rate)
    assert dec == bits + [0] * 6


@pytest.mark.parametrize("rate", PUNCTURED_RATES)
def test_punctured_llr_roundtrip(rate):
    bits = [random.randint(0, 1) for _ in range(64)]
    enc = encode(bits, rate=rate)
    llrs = [5.0 if b == 0 else -5.0 for b in enc]
    dec = viterbi_decode(llrs, rate=rate)
    assert dec == bits


@pytest.mark.parametrize("rate", PUNCTURED_RATES)
def test_punctured_corrects_two_flips(rate):
    # 2 flips lie within the free-distance guarantee of every punctured rate
    bits = [random.randint(0, 1) for _ in range(256)]
    enc = encode(bits, rate=rate)
    rx = enc[:]
    for idx in random.sample(range(len(enc)), 2):
        rx[idx] ^= 1
    dec = viterbi_decode(rx, rate=rate)
    assert dec == bits


@pytest.mark.parametrize("rate", PUNCTURED_RATES)
def test_punctured_length_matches_pattern(rate):
    # transmitted length == number of '1' positions of the cyclic pattern
    bits = [random.randint(0, 1) for _ in range(64)]
    enc = encode(bits, rate=rate)
    pattern = PUNCTURE_PATTERNS[rate]
    full = 2 * 64
    ones = sum(1 for i in range(full) if pattern[i % len(pattern)] == "1")
    assert len(enc) == ones


@pytest.mark.parametrize("rate", PUNCTURED_RATES)
def test_byte_padded_decode(rate):
    # CLI-style streams are packed into whole bytes (up to 7 pad bits)
    bits = [random.randint(0, 1) for _ in range(29 * 8)]
    enc = encode(bits, rate=rate)
    padded = enc + [0] * (8 - len(enc) % 8)
    assert len(padded) % 8 == 0
    dec = _decode_byte_padded(padded, rate)
    assert dec == bits


@pytest.mark.parametrize("rate", ["1/2"] + PUNCTURED_RATES)
def test_pattern_definitions(rate):
    pattern = PUNCTURE_PATTERNS[rate]
    # every pattern starts with "11" and contains only 0/1
    assert set(pattern) <= {"0", "1"}
    assert pattern[:2] == "11"


def test_invalid_rate_rejected():
    with pytest.raises(ValueError):
        encode([0, 1], rate="9/10")
    with pytest.raises(ValueError):
        viterbi_decode([0, 1], rate="9/10")
