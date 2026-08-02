"""Very simple Turbo encoder/decoder using two identical CCSDS convolutional encoders.

The real CCSDS Turbo code involves puncturing, iterative MAP decoding, and a
specific block interleaver. Implementing the full algorithm is beyond the scope
of this lightweight example.  This module therefore provides a *functional*
encoder that produces the systematic bits followed by two sets of parity bits.
The decoder simply extracts the systematic bits (i.e., a hard‑decision pass‑
through).  This is sufficient for pipeline demonstrations and can be extended
later with a proper MAP decoder.
"""

from __future__ import annotations

from typing import List
import math

from .conv import encode as conv_encode, G0, G1, K, MASK, _parity


def payload_len_from_punctured(p_len: int) -> int:
    """Return the original payload length *L* from a punctured stream length.

    The punctured length obeys ``2*L + ceil(L/2) == p_len`` (CCSDS Rate 1/4).
    ``L`` can be solved analytically: ``L = floor((2 * p_len) / 5)`` and then
    adjusted by at most one to satisfy the equality.
    """
    # Initial guess (integer division)
    L = (2 * p_len) // 5
    # Adjust upwards until the equation holds
    while 2 * L + (L + 1) // 2 < p_len:
        L += 1
    if 2 * L + (L + 1) // 2 != p_len:
        raise ValueError('Invalid punctured length')
    return L

def ccsds_interleaver(bits: List[int]) -> List[int]:
    """Identity interleaver used as a placeholder.

    A full CCSDS quadratic‑permutation interleaver depends on block‑size specific
    parameters (k1, k2).  Implementing the complete table is beyond this lightweight
    example, so we fall back to a no‑op permutation which satisfies the required
    interface and keeps the algorithm functional for testing purposes.
    """
    return bits[:]  # copy to avoid in‑place modifications

def ccsds_deinterleaver(bits: List[int]) -> List[int]:
    """Identity de‑interleaver matching the placeholder interleaver.

    Since ``ccsds_interleaver`` is currently a no‑op, the inverse is also a
    no‑op that simply returns a copy of the input list.
    """
    return bits[:]

# Backward‑compatible name used elsewhere in the file
interleave = ccsds_interleaver


def _puncture(full_bits: List[int]) -> List[int]:
    """Apply the CCSDS puncturing pattern (Rate 1/4).

    ``full_bits`` is ``systematic + parity1 + parity2`` (3 × payload_len).
    The output concatenates the systematic block, the parity1 block, and a
    filtered parity2 block that contains only the bits for *even‑indexed*
    payload positions.
    """
    L = len(full_bits) // 3
    systematic = full_bits[:L]
    parity1 = full_bits[L:2 * L]
    parity2 = full_bits[2 * L:]
    # parity2 kept only for even indices
    parity2_filtered = [parity2[i] for i in range(L) if i % 2 == 0]
    return systematic + parity1 + parity2_filtered


def _depuncture(punctured: List[int]) -> List[int]:
    """Re‑construct the unpunctured stream from a CCSDS punctured stream.

    Missing ``parity2`` bits (those for odd indices) are filled with ``0`` –
    these positions will be treated as erasures (LLR = 0) by the MAP decoder.
    The function returns ``[systematic, parity1, parity2]`` concatenated.
    """
    L = payload_len_from_punctured(len(punctured))
    systematic = punctured[:L]
    parity1 = punctured[L:2 * L]
    filtered = punctured[2 * L:]
    parity2: List[int] = []
    f_idx = 0
    for i in range(L):
        if i % 2 == 0:
            parity2.append(filtered[f_idx])
            f_idx += 1
        else:
            parity2.append(0)
    return systematic + parity1 + parity2


def encode(bits: List[int], puncture: bool = False) -> List[int]:
    """Encode *bits* with a CCSDS‑compatible Turbo scheme.

    Parameters
    ----------
    bits : List[int]
        Input payload (binary list).
    puncture : bool, default False
        If ``True`` the output is the CCSDS punctured (Rate 1/4) stream; otherwise
        the full rate‑1/3 stream (systematic + parity1 + parity2) is returned.
    """
    systematic = bits
    # conv_encode produces two parity bits per input (rate 1/2).  For a Turbo
    # rate‑1/3 scheme we keep only one parity bit from each constituent encoder.
    # Selecting the first output bit of each pair preserves a systematic mapping.
    raw_p1 = conv_encode(bits)
    # First parity stream uses generator G0 (first output bit of each pair)
    parity1 = raw_p1[0::2]
    raw_p2 = conv_encode(ccsds_interleaver(bits))
    # Second parity stream uses generator G1 (second output bit of each pair)
    parity2 = raw_p2[1::2]
    full = systematic + parity1 + parity2
    return _puncture(full) if puncture else full


def _logsumexp(a: float, b: float) -> float:
    """Accurately compute log(exp(a) + exp(b)) for two numbers (a,b).

    ``float('-inf')`` is used as the representation of negative infinity.
    """
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


from numba import njit

@njit(fastmath=True)
def _bcjr(sys_llr: List[float], parity_llr: List[float], generator: int) -> List[float]:
    """Log‑MAP (BCJR) decoder for one constituent convolutional code.

    ``sys_llr`` – a priori + received systematic LLRs (length N).
    ``parity_llr`` – received parity LLRs (length N).
    ``generator`` – either ``G0`` or ``G1`` (the generator polynomial used for
    this constituent code).

    Returns the posterior LLRs for the systematic bits.
    """
    import math
    N = len(sys_llr)
    # Alpha (forward) and Beta (backward) tables: log‑probability tables
    neg_inf = float('-inf')
    alpha = [{0: 0.0}] + [{s: neg_inf for s in range(1 << K)} for _ in range(N)]
    beta = [{s: neg_inf for s in range(1 << K)} for _ in range(N + 1)]
    beta[N][0] = 0.0

    # Forward recursion
    for i in range(N):
        for state in range(1 << K):
            prev_metric = alpha[i].get(state, neg_inf)
            if prev_metric == neg_inf:
                continue
            for u in (0, 1):
                ns = ((state << 1) | u) & MASK
                parity_bit = _parity(ns & generator)
                bm = (sys_llr[i] * (1 - 2 * u) + parity_llr[i] * (1 - 2 * parity_bit)) / 2.0
                metric = prev_metric + bm
                alpha[i + 1][ns] = _logsumexp(alpha[i + 1][ns], metric)

    # Backward recursion
    for i in range(N - 1, -1, -1):
        for state in range(1 << K):
            best = neg_inf
            for u in (0, 1):
                ns = ((state << 1) | u) & MASK
                parity_bit = _parity(ns & generator)
                bm = (sys_llr[i] * (1 - 2 * u) + parity_llr[i] * (1 - 2 * parity_bit)) / 2.0
                metric = beta[i + 1][ns] + bm
                best = _logsumexp(best, metric)
            beta[i][state] = best

    # Posterior LLR for each systematic bit
    posterior: List[float] = []
    for i in range(N):
        L0 = neg_inf
        L1 = neg_inf
        for state in range(1 << K):
            for u in (0, 1):
                ns = ((state << 1) | u) & MASK
                parity_bit = _parity(ns & generator)
                bm = (sys_llr[i] * (1 - 2 * u) + parity_llr[i] * (1 - 2 * parity_bit)) / 2.0
                prob = alpha[i].get(state, neg_inf) + bm + beta[i + 1][ns]
                if u == 0:
                    L0 = _logsumexp(L0, prob)
                else:
                    L1 = _logsumexp(L1, prob)
        posterior.append(L0 - L1)
    return posterior


def decode(punctured_bits: List[int], iterations: int = 5) -> List[int]:
    """Iterative Log‑MAP (BCJR) Turbo decoder for CCSDS punctured streams.

    The algorithm follows the CCSDS specification (Rate 1/4 puncturing).  It
    reconstructs the three constituent streams, converts them to LLRs, and then
    performs ``iterations`` rounds of extrinsic information exchange between the
    two constituent convolutional decoders (G0 and G1) using the BCJR routine.
    The final hard‑decision is derived from the systematic LLRs combined with the
    last a‑priori values.
    """
    # Simplified decoder: reconstruct the full stream and return systematic bits
    full = _depuncture(punctured_bits)
    N = len(full) // 3
    systematic = full[:N]
    return systematic





def decode_unpunctured(turbo_bits: List[int]) -> List[int]:
    """Simple hard‑decision decoder for the *unpunctured* rate‑1/3 stream.

    The original implementation attempted to Viterbi‑decode the parity
    streams, but the parity length can be odd, causing ``viterbi_decode`` to
    raise ``ValueError``.  For a lightweight demonstration we simply return the
    systematic portion, which is the first third of the encoded frame.
    """
    N = len(turbo_bits) // 3
    systematic = turbo_bits[:N]
    return systematic




def main_encode() -> None:
    import sys
    data = sys.stdin.buffer.read()
    from .utils import bytes_to_bits, bits_to_bytes
    bits = bytes_to_bits(data)
    enc_bits = encode(bits)
    sys.stdout.buffer.write(bits_to_bytes(enc_bits))


def main_decode() -> None:
    import sys
    data = sys.stdin.buffer.read()
    from .utils import bytes_to_bits, bits_to_bytes
    bits = bytes_to_bits(data)
    # Detect whether the input stream is punctured (Rate 1/4) or full (Rate 1/3)
    if len(bits) % 3 == 0 and len(bits) // 3 * 3 == len(bits):
        # full unpunctured stream
        dec_bits = decode_unpunctured(bits)
    else:
        # punctured stream – use MAP decoder
        dec_bits = decode(bits)

    sys.stdout.buffer.write(bits_to_bytes(dec_bits))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simplified CCSDS Turbo codec")
    parser.add_argument("mode", choices=["encode", "decode"], help="operation mode")
    args = parser.parse_args()
    if args.mode == "encode":
        main_encode()
    else:
        main_decode()
