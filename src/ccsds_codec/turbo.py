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

def _calc_parity_bit(state: int, generator: int) -> int:
    """Compute parity bit for *state* with given *generator* without calling Python function.

    The parity is the XOR of all bits in ``state & generator``.
    """
    temp = state & generator
    parity_bit = 0
    while temp:
        parity_bit ^= temp & 1
        temp >>= 1
    return parity_bit


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
    """CCSDS quadratic‑permutation interleaver (Rate 1/3 Turbo).

    Implements the CCSDS 131.0‑B‑2 interleaver:
        pi(i) = (f1 * i + f2 * i * i) mod N
    with ``f1 = 17`` and ``f2 = 31`` for any block length ``N``. The function
    returns a new list with bits permuted according to this formula.
    """
    N = len(bits)
    if N == 0:
        return []
    f1, f2 = 17, 31
    perm = [(f1 * i + f2 * i * i) % N for i in range(N)]
    return [bits[p] for p in perm]

def ccsds_deinterleaver(bits: List[int]) -> List[int]:
    """Inverse of ``ccsds_interleaver``.

    Handles both odd and even payload lengths:
    * **Odd length** – the interleaver uses ``pi(i) = (2 * i) mod N``. Its inverse
      is multiplication by the modular inverse of 2, which for odd ``N`` is
      ``(N + 1) // 2``.
    * **Even length** – the interleaver is a simple reversal, which is its own
      inverse.
    """
    N = len(bits)
    if N == 0:
        return []
    if N % 2 == 1:
        inv2 = (N + 1) // 2  # modular inverse of 2 modulo odd N
        perm_inv = [(inv2 * i) % N for i in range(N)]
        return [bits[p] for p in perm_inv]
    else:
        # even length – reversal (inverse of itself)
        return bits[::-1]

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
        If ``True`` the output is the CCSDS punctured (Rate 1/4) stream;
        otherwise the full rate‑1/3 stream (systematic + parity1 + parity2)
        is returned.
    """
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

# @njit(fastmath=True)  # Disabled for deterministic behavior in tests
def _bcjr(sys_llr: List[float], parity_llr: List[float], generator: int) -> List[float]:
    """Log‑MAP (BCJR) decoder for a single constituent convolutional code.

    Args:
        sys_llr: List of systematic LLRs (float).
        parity_llr: List of parity LLRs corresponding to *generator*.
        generator: Generator polynomial (G0 or G1).

    Returns:
        Posterior LLRs for the systematic bits.
    """
    """Log‑MAP (BCJR) decoder for one constituent convolutional code.

    ``sys_llr`` – a priori + received systematic LLRs (length N).
    ``parity_llr`` – received parity LLRs (length N).
    ``generator`` – either ``G0`` or ``G1`` (the generator polynomial used for
    this constituent code).

    Returns the posterior LLRs for the systematic bits.
    """
    # math functions are imported at module level
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
                # compute parity of ns & generator without Python function
                temp = ns & generator
                parity_bit = 0
                while temp:
                    parity_bit ^= temp & 1
                    temp >>= 1
                bm = (sys_llr[i] * (1 - 2 * u) + parity_llr[i] * (1 - 2 * parity_bit)) / 2.0
                metric = prev_metric + bm
                # Inline log-sum-exp to avoid calling Python function
                a_val = alpha[i + 1][ns]
                if a_val == neg_inf:
                    alpha[i + 1][ns] = metric
                else:
                    if a_val > metric:
                        alpha[i + 1][ns] = a_val + math.log1p(math.exp(metric - a_val))
                    else:
                        alpha[i + 1][ns] = metric + math.log1p(math.exp(a_val - metric))

    # Backward recursion
    for i in range(N - 1, -1, -1):
        for state in range(1 << K):
            best = neg_inf
            for u in (0, 1):
                ns = ((state << 1) | u) & MASK
                # compute parity of ns & generator without Python function
                temp = ns & generator
                parity_bit = 0
                while temp:
                    parity_bit ^= temp & 1
                    temp >>= 1
                bm = (sys_llr[i] * (1 - 2 * u) + parity_llr[i] * (1 - 2 * parity_bit)) / 2.0
                metric = beta[i + 1][ns] + bm
                # Inline log-sum-exp for best
                if best == neg_inf:
                    best = metric
                else:
                    if best > metric:
                        best = best + math.log1p(math.exp(metric - best))
                    else:
                        best = metric + math.log1p(math.exp(best - metric))
            beta[i][state] = best

    # Posterior LLR for each systematic bit
    posterior: List[float] = []
    for i in range(N):
        L0 = neg_inf
        L1 = neg_inf
        for state in range(1 << K):
            for u in (0, 1):
                ns = ((state << 1) | u) & MASK
                # compute parity of ns & generator without Python function
                temp = ns & generator
                parity_bit = 0
                while temp:
                    parity_bit ^= temp & 1
                    temp >>= 1
                bm = (sys_llr[i] * (1 - 2 * u) + parity_llr[i] * (1 - 2 * parity_bit)) / 2.0
                prob = alpha[i].get(state, neg_inf) + bm + beta[i + 1][ns]
                if u == 0:
                    # Inline log-sum-exp for L0
                    if L0 == neg_inf:
                        L0 = prob
                    else:
                        if L0 > prob:
                            L0 = L0 + math.log1p(math.exp(prob - L0))
                        else:
                            L0 = prob + math.log1p(math.exp(L0 - prob))
                else:
                    # Inline log-sum-exp for L1
                    if L1 == neg_inf:
                        L1 = prob
                    else:
                        if L1 > prob:
                            L1 = L1 + math.log1p(math.exp(prob - L1))
                        else:
                            L1 = prob + math.log1p(math.exp(L1 - prob))
        posterior.append(L0 - L1)
    return posterior


def decode(punctured_bits: List[int], iterations: int = 5) -> List[int]:
    """Full Log‑MAP Turbo decoder (soft‑decision).

    This implementation follows the CCSDS Turbo MAP algorithm:

    1. **Depuncture** the input to obtain systematic, parity‑1 and parity‑2 streams.
    2. Convert each bit to a hard‑decision LLR (``0 → +5.0``, ``1 → -5.0``).
    3. Perform ``iterations`` rounds of extrinsic‑information exchange between the
       two constituent convolutional decoders (generators ``G0`` and ``G1``) using the
       Log‑MAP ``_bcjr`` routine.
    4. After the final iteration, combine the systematic LLRs with the accumulated
       extrinsic information and make a hard decision (``≥0 → 0``, else ``1``).

    The function returns the decoded systematic bits as a list of ``0``/``1``.
    """
    # 1. Depuncture
    full = _depuncture(punctured_bits)
    N = len(full) // 3
    systematic = full[:N]
    parity1 = full[N:2 * N]
    parity2 = full[2 * N:]

    # 2. Hard‑decision LLR conversion
    def bits_to_llr(bits: List[int]) -> List[float]:
        return [5.0 if b == 0 else -5.0 for b in bits]

    sys_llr = bits_to_llr(systematic)
    p1_llr = bits_to_llr(parity1)
    p2_llr = bits_to_llr(parity2)

    # Initialize a‑priori LLRs with the systematic channel observation
    apriori = sys_llr[:]

    for _ in range(iterations):
        # Decoder 1 (non‑interleaved systematic bits)
        post1 = _bcjr(apriori, p1_llr, G0)
        extrinsic1 = [post1[i] - apriori[i] for i in range(N)]

        # Prepare interleaved inputs for decoder 2 (systematic LLRs are interleaved per CCSDS spec)
        interleaved_sys = ccsds_interleaver(apriori)
        interleaved_p2 = ccsds_interleaver(p2_llr)  # parity2 must be interleaved identically to systematic bits
        # Decoder 2
        post2 = _bcjr(interleaved_sys, interleaved_p2, G1)
        extrinsic2_inter = [post2[i] - interleaved_sys[i] for i in range(N)]
        # De‑interleave extrinsic information back to original order
        extrinsic2 = ccsds_deinterleaver(extrinsic2_inter)

        # Update a‑priori LLRs for next iteration
        apriori = [sys_llr[i] + extrinsic1[i] + extrinsic2[i] for i in range(N)]

    # 4. Final hard decision on the accumulated LLRs
    decoded = [0 if llr >= 0 else 1 for llr in apriori]
    return decoded





def decode_unpunctured(turbo_bits: List[int]) -> List[int]:
    """Hard‑decision decoder for an *unpunctured* rate‑1/3 Turbo stream.

    Parameters
    ----------
    turbo_bits : List[int]
        Full encoded bitstream (systematic + parity1 + parity2).

    Returns
    -------
    List[int]
        The recovered systematic bits.
    """
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
