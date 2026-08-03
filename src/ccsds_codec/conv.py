"""CCSDS convolutional encoder/decoder (rate 1/2, constraint length 7).

Implements the convolutional code specified in CCSDS 131.0-B-4 / 101.0-B-4:

* G1 = 1111001 (octal 171) — first output symbol, transmitted uninverted
* G2 = 1011011 (octal 133) — second output symbol, inverted on the channel
  ("symbol inversion on the output path of G2")

The shift register keeps the newest input bit at the LSB, so the generator
polynomials are given in the matching *lsb-current* representation (bit j of
the polynomial taps the input received j steps ago):

* G0 = 0x4F  (bit-reversed 0x79 = 171_8) — first output, not inverted
* G1 = 0x6D  (bit-reversed 0x5B = 133_8) — second output, inverted on channel

This produces exactly the stream expected by gr-satellites' CCSDS receiver
configuration (GNU Radio ``fec.cc_decoder`` with polys ``[79, -109]``).
"""

from __future__ import annotations

from numba import njit

# Generator polynomials in lsb-current representation (LSB = newest input bit):
#   G0 = 0x4F  <-> CCSDS G1 = 171_8 (first output, not inverted)
#   G1 = 0x6D  <-> CCSDS G2 = 133_8 (second output, inverted on the channel)
G0 = 0x4F
G1 = 0x6D
K = 7  # constraint length
MASK = (1 << K) - 1


@njit(fastmath=True)
def _parity(x: int) -> int:
    """Return the parity (XOR of all bits) of ``x`` as 0 or 1."""
    p = 0
    while x:
        p ^= x & 1
        x >>= 1
    return p


def encode(bits: list[int], terminate: bool = False) -> list[int]:
    """Encode a list of input bits to the CCSDS rate-1/2 convolutional code.

    The output order is ``[s0, s1, s0, s1, ...]`` where ``s0`` is the first
    output symbol (CCSDS G1 = 171_8) and ``s1`` is the second output symbol
    (CCSDS G2 = 133_8) inverted, matching the on-air convention used by
    gr-satellites (GNU Radio polys ``[79, -109]``).
    """
    if not bits:
        return []
    # Validate bits are 0 or 1
    for i, b in enumerate(bits):
        if b not in (0, 1):
            raise ValueError(f"Bit at position {i} is not 0 or 1: {b}")
    state = 0
    out: list[int] = []
    for b in bits:
        # shift left, insert new bit at LSB (newest) – we keep newest at LSB
        state = ((state << 1) | (b & 1)) & MASK
        out.append(_parity(state & G0))
        out.append(1 - _parity(state & G1))

    if terminate:
        # flush the shift register by feeding K-1 zeros (tail bits)
        for _ in range(K - 1):
            state = ((state << 1) | 0) & MASK
            out.append(_parity(state & G0))
            out.append(1 - _parity(state & G1))
    return out


# numba disabled for compatibility


def encode_cxx(bits: list[int], terminate: bool = True) -> list[int]:
    """Encode using the C++ ``ViterbiCodec`` algorithm (as found in *gr-satellites*).

    Mirrors the C++ ``ViterbiCodec`` table construction and state update
    (polys ``0x4F``/``0x6D``) and then inverts the second output symbol per the
    CCSDS channel convention. The result is bit-identical to the stream that
    gr-satellites' receiver decodes with GNU Radio ``fec.cc_decoder``
    (polys ``[79, -109]``), including the ``K-1`` zero-flush bits.

    Parameters
    ----------
    bits: List[int]
        Input payload (0/1). Must be non‑empty.
    terminate: bool, default True
        Whether to append the ``K‑1`` flushing zeros (identical to the C++
        encoder's behaviour). Set to ``False`` to obtain the raw 2·len(bits)
        output.
    """
    if not bits:
        raise ValueError("Input bit list must not be empty for C++‑compatible encode")
    for i, b in enumerate(bits):
        if b not in (0, 1):
            raise ValueError(f"Bit at position {i} is not 0 or 1: {b}")

    # ---- 1. Build reversed polynomials (lsb‑current representation) ----
    def rev(poly: int) -> int:
        out = 0
        for _ in range(K):
            out = (out << 1) | (poly & 1)
            poly >>= 1
        return out

    rev_polys = [rev(G0), rev(G1)]

    # ---- 2. Pre‑compute output table (identical to ViterbiCodec::InitializeOutputs) ----
    outputs: list[str] = ["" for _ in range(1 << K)]
    for state in range(1 << K):
        out_bits = []
        for poly in rev_polys:
            tmp_state = state
            tmp_poly = poly
            parity = 0
            for _ in range(K):
                parity ^= (tmp_state & 1) & (tmp_poly & 1)
                tmp_state >>= 1
                tmp_poly >>= 1
            out_bits.append("1" if parity else "0")
        outputs[state] = "".join(out_bits)

    # ---- 3. Encode loop (mirrors C++ Encode) ----
    state = 0
    out: list[int] = []
    for b in bits:
        idx = state | (b << (K - 1))
        sym = outputs[idx]
        out.append(int(sym[0]))
        out.append(1 - int(sym[1]))  # G2 inverted on the channel
        # NextState as per C++: (state >> 1) | (b << (K - 2))
        state = (state >> 1) | (b << (K - 2))
    if terminate:
        for _ in range(K - 1):
            idx = state  # input bit is 0
            sym = outputs[idx]
            out.append(int(sym[0]))
            out.append(1 - int(sym[1]))
            state = state >> 1  # shift in zero
    return out


def decode(soft_bits: list[int]) -> list[int]:
    """Decode soft bits using the Viterbi decoder.

    This wrapper provides a conventional ``decode`` entry point so external
    callers (including the high‑level API) can import ``conv.decode`` directly.
    It simply forwards to the existing Viterbi implementation.
    """
    return viterbi_decode(soft_bits)


def viterbi_decode(soft_bits: list[int]) -> list[int]:
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
            out1 = 1 - _parity(ns & G1)  # G2 is inverted on the channel
            next_state[s][inp] = ns
            output_bits[s][inp] = (out0, out1)

    # Path metrics: large initial value
    INF = 10**9
    path_metric = [INF] * (1 << K)
    path_metric[0] = 0
    # Store predecessor information for traceback
    predecessor = [[-1] * (1 << K) for _ in range(len(soft_bits) // 2 + 1)]
    decoded_bits: list[int] = []

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
