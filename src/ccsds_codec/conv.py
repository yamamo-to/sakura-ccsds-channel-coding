"""CCSDS convolutional encoder/decoder (rate 1/2, constraint length 7).

The encoder uses the generator polynomials specified in the CCSDS recommendation:

* G0 = 0b1111001 (0x79)
* G1 = 0o133  # 0b1011011 per CCSDS spec (0x5B)

Both encoder and Viterbi decoder operate on binary bit lists where the most
significant bit of the shift register corresponds to the newest input bit.
"""

from __future__ import annotations

from numba import njit

# Generator polynomials (7‑bit, MSB corresponds to the newest bit)
G0 = 0o121  # 0b1010001 per CCSDS spec
G1 = 0o133  # 0b1011011 per CCSDS spec
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
    """Encode a list of input bits to a rate‑1/2 convolutional code.

    The output order is ``[s0, s1, s0, s1, ...]`` where ``s0`` and ``s1`` are the
    two systematic parity bits for each input bit.
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
        out.append(_parity(state & G1))

    if terminate:
        # flush the shift register by feeding K-1 zeros (tail bits)
        for _ in range(K - 1):
            state = ((state << 1) | 0) & MASK
            out.append(_parity(state & G0))
            out.append(_parity(state & G1))
    return out


# numba disabled for compatibility


def encode_cxx(bits: list[int], terminate: bool = True) -> list[int]:
    """Encode using the exact algorithm employed by the C++ ``ViterbiCodec``
    (as found in *gr‑satellites*). This reproduces the same bit stream, including
    the ``K‑1`` zero‑flush bits, so that the output matches the reference C++
    implementation.

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
        out.extend(int(ch) for ch in outputs[idx])
        # NextState as per C++: (state >> 1) | (b << (K - 2))
        state = (state >> 1) | (b << (K - 2))
    if terminate:
        for _ in range(K - 1):
            idx = state  # input bit is 0
            out.extend(int(ch) for ch in outputs[idx])
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
            out1 = _parity(ns & G1)
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
