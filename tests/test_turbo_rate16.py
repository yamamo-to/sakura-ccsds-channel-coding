"""Tests for Turbo rate 1/6 (full CCSDS code: G2/G3 constituent polynomials)."""

import random

import numpy as np
import pytest

from ccsds_codec.turbo import (
    GEN,
    GEN_SYS,
    _bcjr_kernel,
    _build_trellis,
    decode,
    encode,
)

RATE16_LENGTHS = [8, 16, 32, 64]


@pytest.mark.parametrize("length", RATE16_LENGTHS)
def test_rate16_roundtrip(length):
    bits = [random.randint(0, 1) for _ in range(length)]
    enc = encode(bits, rate="1/6")
    assert len(enc) == 6 * (length + 4)
    dec = decode(enc, rate="1/6", iterations=5)
    assert dec == bits


def test_rate16_systematic_stride():
    # the systematic payload occupies the first K stride-6 positions
    # (codeword-major layout); the remaining 4 are termination tail bits
    bits = [1, 0, 1, 1, 0, 1, 0, 0]
    enc = encode(bits, rate="1/6")
    assert enc[0::6][: len(bits)] == bits


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
    # length-based auto-detection only recognises the CCSDS standard block
    # lengths; K=16 is non-standard so the rate must be given explicitly
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


def test_bcjr_kernel_channel_matrix_shapes():
    # the Log-MAP kernel consumes a (ncomp, N) channel matrix and returns
    # finite extrinsic/APP LLRs for the information positions
    ns, x, pred0, pred1 = _build_trellis([GEN_SYS, GEN])
    data_len = 8
    n_parity = 2
    N = data_len + 4
    ch = np.zeros((n_parity, N), dtype=np.float64)  # punctured positions 0.0
    la = np.zeros(data_len, dtype=np.float64)
    ext, app = _bcjr_kernel(ch, la, ns, x, pred0, pred1, data_len)
    assert ext.shape == (data_len,)
    assert app.shape == (data_len,)
    assert np.all(np.isfinite(ext))
    assert np.all(np.isfinite(app))


def test_bcjr_kernel_clean_channel_decodes_zeros():
    # a clean all-zero channel (bit 0 -> +1 LLR) must yield positive APP LLRs
    ns, x, pred0, pred1 = _build_trellis([GEN_SYS, GEN])
    data_len = 8
    n_parity = 2
    N = data_len + 4
    ch = np.ones((n_parity, N), dtype=np.float64)
    la = np.zeros(data_len, dtype=np.float64)
    ext, app = _bcjr_kernel(ch, la, ns, x, pred0, pred1, data_len)
    assert np.all(app > 0.0)
