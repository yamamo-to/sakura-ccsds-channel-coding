"""CCSDS pseudo-randomizer (scrambler/descrambler).

Implements the 255-bit pseudo-random sequence of CCSDS 131.0-B-4/5
(sec. 10.4, "Pseudo-Randomizer"; historically sec. 7.4 of
CCSDS 131.0-B-1).  The sequence generator is an 8-stage linear
feedback shift register (LFSR) with characteristic polynomial

    h(x) = x^8 + x^7 + x^5 + x^3 + 1

initialized to the all-ones state (``0xFF``) at the start of each
codeblock or transfer frame.  The sequence repeats every 255 bits.
The first 40 bits of the sequence are::

    1111 1111 0100 1000 0000 1110 1100 0000 1001 1010 ...

(the leftmost bit is the first bit XOR-ed with the data stream).

Because scrambling is an XOR with a fixed sequence, the operation is
its own inverse: ``descramble(scramble(x)) == x`` and applying
``scramble`` twice returns the original data.
"""

from __future__ import annotations

__all__ = ["scramble", "descramble", "SEED", "MASK", "TAPS"]

# LFSR parameters (CCSDS 131.0-B-4 sec. 10.4):
#   * 8-bit shift register, characteristic polynomial x^8+x^7+x^5+x^3+1
#   * seeded to all-ones at the start of every codeblock
#   * output bit = MSB (bit 7)
#   * feedback bit = parity of bits {7, 4, 2, 0} of the register
SEED = 0xFF  # all-ones state
MASK = 0xFF  # 8-bit register mask
TAPS = (7, 4, 2, 0)  # feedback taps (zero-indexed, counted from the LSB)


def _lfsr_next(state: int) -> tuple[int, int]:
    """Advance the 8-bit LFSR by one step.

    Args:
        state: current 8-bit register value (0..255).

    Returns:
        ``(output_bit, new_state)`` where ``output_bit`` is the MSB of
        ``state`` (bit 7) and ``new_state`` is the shifted register with
        the feedback parity of bits {7, 4, 2, 0} fed into the LSB.
    """
    out = (state >> 7) & 1
    fb = 0
    for t in TAPS:
        fb ^= (state >> t) & 1
    new_state = ((state << 1) & MASK) | fb
    return out, new_state


def scramble(bits: list[int]) -> list[int]:
    """Scramble a list of bits with the CCSDS pseudo-random sequence.

    Args:
        bits: bit list (each element 0 or 1), MSB-first per byte.

    Returns:
        The XOR of ``bits`` with the pseudo-random sequence.  The LFSR
        is re-seeded to the all-ones state at the start of the call,
        matching the "one sequence per codeblock" CCSDS requirement.
    """
    state = SEED
    out_bits: list[int] = []
    for bit in bits:
        lfsr_bit, state = _lfsr_next(state)
        out_bits.append(bit ^ lfsr_bit)
    return out_bits


def descramble(bits: list[int]) -> list[int]:
    """Descramble bits scrambled by :func:`scramble`.

    The CCSDS randomizer is its own inverse, so this is an alias for
    :func:`scramble`.
    """
    return scramble(bits)
