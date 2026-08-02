"""Extended tests for CCSDS randomizer (scrambler/descrambler)"""

import random
from ccsds_codec.randomizer import scramble, descramble

def test_identity_various_lengths():
    for length in [0, 1, 5, 16, 31, 64, 127, 128, 255]:
        bits = [random.randint(0, 1) for _ in range(length)]
        assert descramble(scramble(bits)) == bits

def test_repeatability():
    # Apply scramble twice – should return original (self‑inverse)
    bits = [random.randint(0, 1) for _ in range(100)]
    once = scramble(bits)
    twice = scramble(once)
    assert twice == bits
