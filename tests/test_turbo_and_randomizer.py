"""Extended tests for Turbo codec and Randomizer.

* Verify that the un‑punctured Turbo encoder followed by the simple `decode_unpunctured` recovers the original systematic bits.
* Verify that the punctured encoder followed by `decode` (which internally depunctures) also recovers the systematic bits.
* Verify that the LFSR‑based randomizer `scramble`/`descramble` is self‑inverse.
"""

import os
import random
import pytest

from ccsds_codec import turbo, randomizer

def _random_bits(length: int) -> list[int]:
    random.seed(0)
    return [random.randint(0, 1) for _ in range(length)]

def test_turbo_unpunctured_roundtrip():
    bits = _random_bits(64)
    encoded = turbo.encode(bits, puncture=False)
    assert len(encoded) == 3 * len(bits)
    decoded = turbo.decode_unpunctured(encoded)
    assert decoded == bits

def test_turbo_punctured_roundtrip():
    bits = _random_bits(60)
    encoded = turbo.encode(bits, puncture=True)
    decoded = turbo.decode(encoded)
    assert decoded == bits

def test_randomizer_self_inverse():
    data = os.urandom(16)
    from ccsds_codec.utils import bytes_to_bits
    bits = bytes_to_bits(data)
    scrambled = randomizer.scramble(bits)
    descrambled = randomizer.descramble(scrambled)
    assert descrambled == bits
    double = randomizer.scramble(randomizer.scramble(bits))
    assert double == bits
