"""Tests for Turbo rate 1/6 (full CCSDS code: G2/G3 constituent polynomials)."""

import random

import numpy as np
import pytest

from ccsds_codec.turbo import GEN, decode, encode, _bcjr_kernel, _bcjr_multi_kernel

RATE16_LENGTHS = [8, 16, 32, 64]


@pytest.mark.parametrize("length", RATE16_LENGTHS)
def test_rate16_roundtrip(length):
    bits = [random.randint(0, 1) for _ in range(length)]
    enc = encode(bits, rate="1/6")
    assert len(enc) == 6 * length + 20
    dec = decode(enc, rate="1/6", iterations=5)
    assert dec == bits


def test_rate16_systematic_prefix():
    # the first L bits of the stream are the systematic payload
    bits = [1, 0, 1, 1, 0]
    enc = encode(bits, rate="1/6")
    assert enc[:5] == bits


@pytest.mark.parametrize("length", RATE16_LENGTHS)
def test_rate16_corrects_flips(length):
    # the full rate-1/6 code corrects several systematic bit flips
    bits = [random.randint(0, 1) for _ in range(length)]
    enc = encode(bits, rate="1/6")
    rx = enc[:]
    for idx in random.sample(range(length), min(4, length)):
        rx[idx] ^= 1
    dec = decode(rx, rate="1/6", iterations=8)
    assert dec == bits


def test_rate16_requires_explicit_rate():
    # length-based detection cannot distinguish rate-1/6 frames: without the
    # explicit rate the stream is not recognized as rate-1/6
    bits = [random.randint(0, 1) for _ in range(16)]
    enc = encode(bits, rate="1/6")
    with pytest.raises(ValueError):
        decode(enc)


def test_rate16_invalid_length():
    with pytest.raises(ValueError):
        decode([0] * 30, rate="1/6")
    with pytest.raises(ValueError):
        decode([0] * 10, rate="1/6")


def test_rate16_empty():
    assert encode([], rate="1/6") == []
    assert decode([], rate="1/6") == []


def test_bcjr_multi_kernel_matches_single():
    # the multi-parity kernel with one generator equals the 1-parity kernel
    sys_llr = np.array([1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0])
    par_llr = np.array([1.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0])
    single = _bcjr_kernel(sys_llr, par_llr, GEN, 5)
    multi = _bcjr_multi_kernel(
        sys_llr, par_llr.reshape(1, -1), np.array([GEN], dtype=np.int64), 5
    )
    assert np.allclose(single, multi)
