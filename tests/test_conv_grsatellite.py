import numpy as np
import pytest

from ccsds_codec.conv import encode_cxx, viterbi_decode


@pytest.mark.parametrize("length", [1, 10, 100, 255])
def test_encode_decode_roundtrip(length):
    # random bits
    bits = np.random.randint(0, 2, size=length).tolist()
    # encode using C++ compatible encoder (includes termination)
    enc = encode_cxx(bits, terminate=True)
    # decode using Viterbi decoder
    dec = viterbi_decode(enc)
    # Compare only the original bits (decoder may include termination zeros)
    assert dec[: len(bits)] == bits
