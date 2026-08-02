"""Known‑vector tests for the CCSDS convolutional encoder (Rate 1/2, K=7)."""

import pytest
from ccsds_codec.conv import encode, G0, G1

def test_known_vector():
    # Input pattern 0b10101010 (bits MSB‑first)
    bits = [1, 0, 1, 0, 1, 0, 1, 0]
    # Expected output computed from the CCSDS spec (generator polynomials G0=0o121, G1=0o133)
    expected = [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    assert encode(bits) == expected

def test_generator_constants():
    assert G0 == 0o121
    assert G1 == 0o133
