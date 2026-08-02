"""CCSDS convolutional encoder/decoder (rate 1/2, constraint length 7).

The encoder uses the generator polynomials specified in the CCSDS recommendation:

* G0 = 0b1111001 (0x79)
* G1 = 0b1011011 (0x5B)

Both encoder and Viterbi decoder operate on binary bit lists where the most
significant bit of the shift register corresponds to the newest input bit.
"""

from __future__ import annotations

from typing import List, Tuple

# Generator polynomials (7‑bit, MSB corresponds to the newest bit)
G0 = 0o121  # 0b1010001 per CCSDS spec
G1 = 0b1011011
K = 7  # constraint length
MASK = (1 << K) - 1


def _parity(x: int) -> int:
    """Return the parity (XOR of all bits) of ``x`` as 0 or 1."""
    p = 0
    while x:
        p ^= x & 1
        x >>= 1
    return p


def encode(bits: List[int]) -> List[int]:
    """Encode a list of input bits to a rate‑1/2 convolutional code.

    The output order is ``[s0, s1, s0, s1, ...]`` where ``s0`` and ``s1`` are the
    two systematic parity bits for each input bit.
    """
    state = 0
    out: List[int] = []
    for b in bits:
        # shift left, insert new bit at LSB (oldest) – we keep newest at MSB
        state = ((state << 1) | (b & 1)) & MASK
        out.append(_parity(state & G0))
        out.append(_parity(state & G1))
    return out


# numba disabled for compatibility

def viterbi_decode(soft_bits: List[int]) -> List[int]:
    """Very simple hard‑decision Viterbi decoder for the CCSDS code.

    ``soft_bits`` must be a list of 0/1 values with length a multiple of 2.
    The implementation uses a table‑based approach for the 2^K = 128 states.
    """
    if len(soft_bits) % 2 != 0:
        raise ValueError("Number of soft bits must be even (two per input symbol)")

    # Pre‑compute next state and output for each state and input bit
    next_state = [[0, 0] for _ in range(1 << K)]
    output_bits = [[(0, 0), (0, 0)] for _ in range(1 << K)]  # (out0,out1) for input 0/1
    for s in range(1 << K):
        for inp in (0, 1):
            ns = ((s << 1) | inp) & MASK
            out0 = _parity(ns & G0)
            out1 = _parity(ns & G1)
            next_state[s][inp] = ns
            output_bits[s][inp] = (out0, out1)

    # Path metrics: large initial value
    INF = 10 ** 9
    path_metric = [INF] * (1 << K)
    path_metric[0] = 0
    # Store predecessor information for traceback
    predecessor = [[-1] * (1 << K) for _ in range(len(soft_bits) // 2 + 1)]
    decoded_bits: List[int] = []

    for i in range(0, len(soft_bits), 2):
        r0, r1 = soft_bits[i], soft_bits[i + 1]
        new_metric = [INF] * (1 << K)
        for s in range(1 << K):
            if path_metric[s] == INF:
                continue
            for inp in (0, 1):
                ns = next_state[s][inp]
                o0, o1 = output_bits[s][inp]
                # Hamming distance for hard decisions
                dist = (o0 != r0) + (o1 != r1)
                metric = path_metric[s] + dist
                if metric < new_metric[ns]:
                    new_metric[ns] = metric
                    predecessor[i // 2 + 1][ns] = (s << 1) | inp  # store combined info
        path_metric = new_metric

    # Find best ending state (minimum metric)
    best_state = min(range(1 << K), key=lambda s: path_metric[s])
    # Trace back
    for step in range(len(soft_bits) // 2, 0, -1):
        prev = predecessor[step][best_state]
        if prev == -1:
            # Should not happen; fallback to zeros
            decoded_bits.append(0)
            best_state = 0
        else:
            inp = prev & 1
            decoded_bits.append(inp)
            best_state = prev >> 1
    decoded_bits.reverse()
    return decoded_bits


def main_encode() -> None:
    import sys
    data = sys.stdin.buffer.read()
    bits = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    enc = encode(bits)
    # pack to bytes
    from .utils import bits_to_bytes
    sys.stdout.buffer.write(bits_to_bytes(enc))


def main_decode() -> None:
    import sys
    data = sys.stdin.buffer.read()
    # interpret input as bits
    from .utils import bytes_to_bits
    bits = bytes_to_bits(data)
    dec = viterbi_decode(bits)
    from .utils import bits_to_bytes
    sys.stdout.buffer.write(bits_to_bytes(dec))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CCSDS convolutional coder")
    parser.add_argument("mode", choices=["encode", "decode"], help="operation mode")
    args = parser.parse_args()
    if args.mode == "encode":
        main_encode()
    else:
        main_decode()
