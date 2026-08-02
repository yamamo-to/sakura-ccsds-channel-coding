"""CCSDS randomizer (scrambler/descrambler).

The CCSDS standard uses a self‑synchronizing LFSR of length 7 with the polynomial
``x^7 + x^6 + 1``. The seed is all ones (0x7F). The same algorithm is used for
scrambling and descrambling because the operation is its own inverse.
"""

from .utils import bytes_to_bits, bits_to_bytes

POLY_TAPS = (6, 5)  # taps for x^7 + x^6 + 1 (zero‑based indexing)
SEED = 0x7F  # 0b1111111


def _lfsr_next(state: int) -> (int, int):
    """Return (output_bit, new_state) for one LFSR step.

    ``state`` holds the 7‑bit register value.
    """
    # XOR the tapped bits
    tap = ((state >> POLY_TAPS[0]) ^ (state >> POLY_TAPS[1])) & 1
    out = tap
    new_state = ((state << 1) & 0x7F) | tap
    return out, new_state


def scramble(bits: list[int]) -> list[int]:
    """Scramble a list of bits using the CCSDS LFSR.

    The operation is its own inverse; calling ``scramble`` again descrambles.
    """
    state = SEED
    out_bits: list[int] = []
    for bit in bits:
        lfsr_bit, state = _lfsr_next(state)
        out_bits.append(bit ^ lfsr_bit)
    return out_bits


def descramble(bits: list[int]) -> list[int]:
    """Alias for ``scramble`` – the algorithm is symmetric."""
    return scramble(bits)


def main() -> None:
    import sys
    data = sys.stdin.buffer.read()
    bits = bytes_to_bits(data)
    scrambled = scramble(bits)
    sys.stdout.buffer.write(bits_to_bytes(scrambled))


if __name__ == "__main__":
    main()
