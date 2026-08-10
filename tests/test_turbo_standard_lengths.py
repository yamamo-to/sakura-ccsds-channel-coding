'''Tests for CCSDS Turbo codec at all standard information block lengths.

This file extends the existing test suite by exercising the interleaver,
encoding/decoding round‑trip, stream‑length formula and rate auto‑detection
for each of the five CCSDS standard block lengths defined in
``STANDARD_K`` (1784, 3568, 7136, 8920, 16384 bits).

The tests use deterministic bit patterns (alternating 0/1) to avoid any
random‑seed dependencies and run the decoder with a single Log‑MAP iteration
(which is sufficient for a clean channel and keeps the total runtime well
under the required ~60 s budget).
'''

import pytest

from ccsds_codec.turbo import (
    STANDARD_K,
    NCOMP,
    encode,
    decode,
    _detect_rate_k,
    ccsds_interleaver,
    ccsds_deinterleaver,
)


@pytest.mark.parametrize("K", STANDARD_K)
def test_interleaver_is_permutation_and_inverse(K):
    # deterministic payload (must be multiple of 8)
    bits = list(range(K))
    inter = ccsds_interleaver(bits)
    # permutation property
    assert len(set(inter)) == K
    assert sorted(inter) == sorted(bits)
    # inverse property
    assert ccsds_deinterleaver(inter) == bits
    # also interleaver after deinterleaver yields original
    assert ccsds_interleaver(ccsds_deinterleaver(bits)) == bits


@pytest.mark.parametrize("K", STANDARD_K)
@pytest.mark.parametrize("rate", ["1/3", "1/6"])
def test_encode_decode_roundtrip_standard_lengths(K, rate):
    bits = [i % 2 for i in range(K)]
    enc = encode(bits, rate=rate)
    assert len(enc) == NCOMP[rate] * (K + 4)
    dec = decode(enc, rate=rate, iterations=1)
    assert dec == bits


@pytest.mark.parametrize("K", STANDARD_K)
@pytest.mark.parametrize("rate", list(NCOMP.keys()))
def test_detect_rate_k_for_standard_lengths(K, rate):
    stream_len = NCOMP[rate] * (K + 4)
    detected_rate, detected_K = _detect_rate_k(stream_len)
    assert detected_rate == rate
    assert detected_K == K
