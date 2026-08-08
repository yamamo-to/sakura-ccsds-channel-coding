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

import numpy as np
from numba import njit

__all__ = [
    "G0",
    "G1",
    "K",
    "MASK",
    "PUNCTURE_PATTERNS",
    "encode",
    "encode_cxx",
    "decode",
    "viterbi_decode",
    "decode_byte_padded",
]

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


# Puncturing patterns for CCSDS convolutional code rates (binary strings).
# "1" = transmit bit, "0" = omit (puncture).
# Patterns are taken from the CCSDS specification / GNU Radio examples.
PUNCTURE_PATTERNS: dict[str, str] = {
    "1/2": "11",
    "2/3": "1101",
    "3/4": "110110",
    "5/6": "1101100110",
    "7/8": "11010101100110",
}


def _validate_bits(bits: list[int]) -> None:
    """Raise :class:`ValueError` unless every element of *bits* is 0 or 1."""
    for i, b in enumerate(bits):
        if b not in (0, 1):
            raise ValueError(f"Bit at position {i} is not 0 or 1: {b}")


def encode(bits: list[int], terminate: bool = False, rate: str = "1/2") -> list[int]:
    """Encode *bits* with the CCSDS convolutional code.

    ``rate`` controls the puncturing applied after the basic rate‑1/2 encoding.
    Supported rates are ``"1/2"``, ``"2/3"``, ``"3/4"``, ``"5/6"`` and ``"7/8"``.
    The ``terminate`` flag adds ``K‑1`` zero‑tail bits before puncturing (as in
    the original rate‑1/2 encoder).  Puncturing is performed on the full output
    sequence (including any tail bits) using the pattern defined in
    :data:`PUNCTURE_PATTERNS`.
    """
    if not bits:
        return []
    _validate_bits(bits)
    state = 0
    out: list[int] = []
    for b in bits:
        state = ((state << 1) | (b & 1)) & MASK
        out.append(_parity(state & G0))
        out.append(1 - _parity(state & G1))
    if terminate:
        for _ in range(K - 1):
            state = ((state << 1) | 0) & MASK
            out.append(_parity(state & G0))
            out.append(1 - _parity(state & G1))
    # Apply puncturing pattern if needed
    pattern = PUNCTURE_PATTERNS.get(rate)
    if pattern is None:
        raise ValueError(f"Unsupported convolutional code rate: {rate}")
    if pattern == "11":
        return out
    punctured: list[int] = []
    pat_len = len(pattern)
    for i, bit in enumerate(out):
        if pattern[i % pat_len] == "1":
            punctured.append(bit)
    return punctured


# --- Trellis tables and Viterbi kernels (numba JIT, per AGENTS.md §4.2) ---


@njit(fastmath=True)
def _build_tables():
    """Precompute the 2^K-state trellis tables.

    Returns three (2^K, 2) integer arrays:

    * ``next_state[s, inp]`` — next state after input ``inp`` in state ``s``
    * ``out0[s, inp]`` — first output symbol (G1 = 171_8, not inverted)
    * ``out1[s, inp]`` — second output symbol (G2 = 133_8, inverted on channel)
    """
    n_states = 1 << K
    next_state = np.empty((n_states, 2), dtype=np.int64)
    out0 = np.empty((n_states, 2), dtype=np.int64)
    out1 = np.empty((n_states, 2), dtype=np.int64)
    for s in range(n_states):
        for inp in (0, 1):
            ns = ((s << 1) | inp) & MASK
            next_state[s, inp] = ns
            out0[s, inp] = _parity(ns & G0)
            out1[s, inp] = 1 - _parity(ns & G1)
    return next_state, out0, out1


@njit(fastmath=True)
def _viterbi_hard_kernel(rx, next_state, out0, out1):
    """Hard-decision Viterbi kernel (Hamming-distance metric).

    A received symbol equal to -1 marks an erasure (a punctured position);
    erased symbols contribute 0 to the distance.
    """
    n = rx.shape[0] // 2
    n_states = next_state.shape[0]
    INF = np.int64(1 << 30)
    path_metric = np.full(n_states, INF, dtype=np.int64)
    path_metric[0] = 0
    # predecessor[step, state] = (prev_state << 1) | inp
    predecessor = np.full((n + 1, n_states), -1, dtype=np.int64)
    for i in range(n):
        r0 = rx[2 * i]
        r1 = rx[2 * i + 1]
        new_metric = np.full(n_states, INF, dtype=np.int64)
        for s in range(n_states):
            pm = path_metric[s]
            if pm == INF:
                continue
            for inp in (0, 1):
                ns = next_state[s, inp]
                d0 = 0 if r0 == -1 else (out0[s, inp] != r0)
                d1 = 0 if r1 == -1 else (out1[s, inp] != r1)
                metric = pm + d0 + d1
                if metric < new_metric[ns]:
                    new_metric[ns] = metric
                    predecessor[i + 1, ns] = (s << 1) | inp
        path_metric = new_metric
    best_state = 0
    best_m = path_metric[0]
    for s in range(1, n_states):
        if path_metric[s] < best_m:
            best_m = path_metric[s]
            best_state = s
    decoded = np.empty(n, dtype=np.int64)
    state = best_state
    for step in range(n, 0, -1):
        prev = predecessor[step, state]
        if prev == -1:
            decoded[step - 1] = 0
            state = 0
        else:
            decoded[step - 1] = prev & 1
            state = prev >> 1
    return decoded


@njit(fastmath=True)
def _viterbi_llr_kernel(rx, next_state, out0, out1):
    """Soft-decision Viterbi kernel (LLR metric, maximized).

    ``rx`` holds log-likelihood ratios with the convention: positive value =
    likelihood of bit 0, negative value = likelihood of bit 1 (AGENTS.md §2).
    """
    n = rx.shape[0] // 2
    n_states = next_state.shape[0]
    NEG_INF = -1e300
    path_metric = np.full(n_states, NEG_INF, dtype=np.float64)
    path_metric[0] = 0.0
    predecessor = np.full((n + 1, n_states), -1, dtype=np.int64)
    for i in range(n):
        llr0 = rx[2 * i]
        llr1 = rx[2 * i + 1]
        new_metric = np.full(n_states, NEG_INF, dtype=np.float64)
        for s in range(n_states):
            pm = path_metric[s]
            if pm == NEG_INF:
                continue
            for inp in (0, 1):
                ns = next_state[s, inp]
                m0 = llr0 if out0[s, inp] == 0 else -llr0
                m1 = llr1 if out1[s, inp] == 0 else -llr1
                metric = pm + m0 + m1
                if metric > new_metric[ns]:
                    new_metric[ns] = metric
                    predecessor[i + 1, ns] = (s << 1) | inp
        path_metric = new_metric
    best_state = 0
    best_m = path_metric[0]
    for s in range(1, n_states):
        if path_metric[s] > best_m:
            best_m = path_metric[s]
            best_state = s
    decoded = np.empty(n, dtype=np.int64)
    state = best_state
    for step in range(n, 0, -1):
        prev = predecessor[step, state]
        if prev == -1:
            decoded[step - 1] = 0
            state = 0
        else:
            decoded[step - 1] = prev & 1
            state = prev >> 1
    return decoded


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
    _validate_bits(bits)

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


def decode(soft_bits: list[int], rate: str = "1/2") -> list[int]:
    """Decode soft bits using the Viterbi decoder.

    This wrapper provides a conventional ``decode`` entry point so external
    callers (including the high‑level API) can import ``conv.decode`` directly.
    It simply forwards to the existing Viterbi implementation.
    """
    return viterbi_decode(soft_bits, rate)


def _depuncture(rx: np.ndarray, pattern: str) -> np.ndarray:
    """Reinsert erasures at punctured positions of the rate-1/2 stream.

    ``rx`` is the received (punctured) symbol stream and ``pattern`` the
    puncturing pattern applied by the encoder (starting at position 0).
    Integer input uses -1 as the erasure marker; float (LLR) input uses 0.0,
    which is neutral for the LLR metric.
    """
    if len(rx) == 0:
        return rx
    pat_len = len(pattern)
    ones = pattern.count("1")
    # Smallest even full length whose cyclic pattern contains len(rx) ones.
    # The ones-count strictly increases with every 2 positions for all
    # CCSDS puncturing patterns, so the match is unique when it exists.
    L = 2 * max(1, (len(rx) * pat_len) // (2 * ones))
    while True:
        full, rem = divmod(L, pat_len)
        cnt = full * ones + pattern[:rem].count("1")
        if cnt == len(rx):
            break
        if cnt > len(rx):
            raise ValueError(
                f"Invalid punctured stream length {len(rx)} for pattern {pattern!r}"
            )
        L += 2
    is_int = rx.dtype.kind in "iu"
    out = np.empty(L, dtype=np.int64 if is_int else np.float64)
    fill = -1 if is_int else 0.0
    j = 0
    for i in range(L):
        if pattern[i % pat_len] == "1":
            out[i] = rx[j]
            j += 1
        else:
            out[i] = fill
    return out


def viterbi_decode(
    soft_bits: list[int] | np.ndarray, rate: str = "1/2"
) -> list[int]:
    """Viterbi decoder for the CCSDS convolutional code (hard or soft input).

    ``soft_bits`` is the received symbol stream at code rate ``rate``
    (``"1/2"``, ``"2/3"``, ``"3/4"``, ``"5/6"`` or ``"7/8"``).  Integer
    inputs (0/1) are decoded with a Hamming-distance (hard-decision) metric;
    float inputs are treated as LLRs (positive = likelihood of 0, negative =
    likelihood of 1, per AGENTS.md §2) and decoded with a maximized LLR
    metric.  For punctured rates the stream is first depunctured by inserting
    erasures at the omitted positions.

    The trellis kernels are JIT-compiled with numba (``@njit(fastmath=True)``).
    """
    pattern = PUNCTURE_PATTERNS.get(rate)
    if pattern is None:
        raise ValueError(f"Unsupported convolutional code rate: {rate}")
    arr = np.asarray(soft_bits)
    if rate != "1/2":
        arr = _depuncture(arr, pattern)
    if len(arr) % 2 != 0:
        raise ValueError("Number of soft bits must be even (two per input symbol)")

    next_state, out0, out1 = _build_tables()
    if arr.dtype.kind in "iu":
        decoded = _viterbi_hard_kernel(arr.astype(np.int64), next_state, out0, out1)
    else:
        decoded = _viterbi_llr_kernel(arr.astype(np.float64), next_state, out0, out1)
    return decoded.tolist()


def decode_byte_padded(soft_bits: list[int], rate: str) -> list[int]:
    """Decode a byte-padded punctured stream (CLI helper).

    The CLI packs the encoded stream into whole bytes, so up to 7 trailing
    padding bits may follow the real code stream.  The true length is the
    unique candidate ``T`` in ``[len-7, len]`` that depunctures to a full
    length which is a multiple of 16 (byte-aligned input bits yield a whole
    number of decoded bytes).  Candidates whose depuncture is invalid or
    whose full length is not byte-clean are skipped.
    """
    pattern = PUNCTURE_PATTERNS[rate]
    n = len(soft_bits)
    for k in range(8):
        cand = soft_bits[: n - k] if k else soft_bits
        try:
            dep = _depuncture(np.asarray(cand), pattern)
        except ValueError:
            continue
        if len(dep) % 16 == 0:
            return viterbi_decode(cand, rate=rate)
    raise ValueError(
        f"Unrecognized {rate} stream: no valid length within the last 8 bits"
    )
