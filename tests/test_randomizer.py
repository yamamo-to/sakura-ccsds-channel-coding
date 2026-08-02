"""Tests for CCSDS randomizer (scrambler)"""

import random
from ccsds_codec.randomizer import scramble, descramble

def test_scramble_descramble_identity():
    bits = [random.randint(0, 1) for _ in range(50)]
    scrambled = scramble(bits)
    descrambled = descramble(scrambled)
    assert descrambled == bits
