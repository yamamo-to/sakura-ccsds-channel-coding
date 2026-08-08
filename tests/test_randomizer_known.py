"""Known-vector tests for the CCSDS pseudo-randomizer (CCSDS 131.0-B-4/5 sec. 10.4)."""

import pytest
from ccsds_codec.randomizer import scramble, descramble
from ccsds_codec.utils import bytes_to_bits, bits_to_bytes


def bits_from_hex(hex_str: str) -> list[int]:
    data = bytes.fromhex(hex_str)
    return bytes_to_bits(data)


def test_known_vector_0x55():
    # CCSDS 255-bit pseudo-randomizer, first 8 sequence bits = 1111 1111 (0xFF).
    # 0x55 (01010101) ^ 0xFF = 0xAA (10101010).
    plain_bits = bits_from_hex('55')
    scrambled = scramble(plain_bits)
    assert bits_to_bytes(scrambled) == bytes.fromhex('aa')


def test_known_sequence_first_40_bits():
    # The standard (CCSDS 131.0-B-4/5 sec. 10.4.3) specifies the first 40 bits
    # of the pseudo-random sequence (scrambling an all-zero stream):
    #   1111 1111 0100 1000 0000 1110 1100 0000 1001 1010 ...
    expected = bytes.fromhex('ff 48 0e c0 9a')
    zeros = bytes_to_bits(b'\x00' * 16)
    scrambled = bits_to_bytes(scramble(zeros))
    assert scrambled[:5] == expected


def test_known_sequence_period():
    # The sequence repeats after 255 bits: scrambling 256 zero bits must give
    # the same first bit at position 0 and position 255 (wraparound).
    seq = scramble(bytes_to_bits(b'\x00' * 32))
    assert seq[0] == seq[255]
    # byte 9 (bits 72..79) of the 256-bit scrambled stream equals 0x2c, the
    # 10th byte of the generated sequence.
    assert bits_to_bytes(seq)[9] == 0x2c


def test_self_inverse():
    # Any random payload should be recovered after double scrambling
    data = bytes.fromhex('DEADBEEF')
    bits = bytes_to_bits(data)
    assert descramble(scramble(bits)) == bits
