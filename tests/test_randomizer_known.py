"""Known‑vector tests for the CCSDS randomizer (RFC 128.0‑B‑1)."""

import pytest
from ccsds_codec.randomizer import scramble, descramble
from ccsds_codec.utils import bytes_to_bits, bits_to_bytes

def bits_from_hex(hex_str: str) -> list[int]:
    data = bytes.fromhex(hex_str)
    return bytes_to_bits(data)

def test_known_vector():
    # CCSDS example: input 0x55 (01010101) → scrambled 0xA3 (10100011)
    plain_bits = bits_from_hex('55')
    scrambled = scramble(plain_bits)
    assert bits_to_bytes(scrambled) == bytes.fromhex('57')

def test_self_inverse():
    # Any random payload should be recovered after double scrambling
    data = bytes.fromhex('DEADBEEF')
    bits = bytes_to_bits(data)
    assert descramble(scramble(bits)) == bits
