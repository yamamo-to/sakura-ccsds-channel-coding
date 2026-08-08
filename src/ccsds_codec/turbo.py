"""CCSDS Turbo encoder/decoder (rates 1/3, 1/4, 1/2, 1/6; QPP interleaver; iterative Log-MAP).

Implements the Turbo code described in ``docs/CCSDS_Turbo_Spec.md`` (a summary
of CCSDS 131.0-B-4 §3 / §5 / §6):

* **Constituent code** – constraint length ``K = 5`` with feedback polynomial
  ``g0 = 10011_2 (23_8)`` and forward polynomial ``g1 = 11011_2 (33_8)``.
  Following the spec's BCJR sketch (docs §2.4) the transitions are computed
  in the lsb-current feed-forward form ``ns = (state << 1 | u) & MASK`` with
  parity ``parity(ns & g1)``, used identically for both constituent encoders.
  The rate-1/6 constituent codes additionally use the forward polynomials
  ``g2 = 10101_2 (25_8)`` and ``g3 = 11111_2 (37_8)`` (CCSDS 131.0-B-4 §3.3.1).
* **Interleaver** – quadratic-permutation π(i) = (f1·i + f2·i²) mod K
  (docs §2.2), **not** self-inverse. Parameters are chosen generically
  (``f1 = 1``, ``f2 = lcm(rad(K), 4 if 4|K else 1)``), which is a valid
  permutation for every block length and covers the CCSDS block lengths
  1784 / 3568 / 7136 / 8920 / 16384.
* **Puncturing** – rate-1/4 keeps systematic + parity1 + the even-indexed
  parity2 bits (docs §2.3). The punctured frame additionally carries the 4
  flush (termination) parity bits of the first constituent so the decoder
  knows the terminal state (CCSDS §3.2.3).  Rate-1/6 is the full unpunctured
  code: systematic + parity1/2/3 of the first constituent + parity1/3 of the
  second, plus the flush parity bits of both constituents.
* **Decoder** – iterative Log-MAP (BCJR) per docs §2.4–2.5, numerically
  stable through ``np.logaddexp`` (no exp overflow/underflow).  LLR
  convention: positive value = likelihood of bit 0, negative = likelihood of
  bit 1 (AGENTS.md §2); the internal scale is ±1.0 (docs §6.2.2).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .conv import _parity

# Constituent-code constants (CCSDS 131.0-B-4 §3.3.1; docs/CCSDS_Turbo_Spec.md):
#   g1 = 11011_2 = 33_8, K = 5  →  4-bit shift register, states 0..15
GEN = 0x1B  # forward polynomial in lsb-current representation
MASK = 0xF  # (1 << (K - 1)) - 1 for K = 5
TAIL = 4  # K - 1 flush (termination) bits per constituent code

# Rate-1/6 forward polynomials (CCSDS 131.0-B-4 §3.3.1, Table 3-1):
#   G2 = 10101_2 = 25_8, G3 = 11111_2 = 37_8
# Both are bit-symmetric, so the lsb-current representation equals the
# standard (msb-first) binary form.
GEN2 = 0x15
GEN3 = 0x1F

# LLR scale (docs §6.2.2): bit 0 → +1.0, bit 1 → -1.0
LLR_0 = 1.0
LLR_1 = -1.0


def payload_len_from_punctured(p_len: int) -> int:
    """Return the original payload length *L* from a punctured stream length.

    The punctured (rate‑1/4) length obeys ``2*L + ceil(L/2) == p_len``
    (docs/CCSDS_Turbo_Spec.md §2.3).  ``L`` can be solved analytically:
    ``L = floor((2 * p_len) / 5)`` and then adjusted by at most one to
    satisfy the equality.
    """
    # Initial guess (integer division)
    L = (2 * p_len) // 5
    # Adjust upwards until the equation holds
    while 2 * L + (L + 1) // 2 < p_len:
        L += 1
    if 2 * L + (L + 1) // 2 != p_len:
        raise ValueError('Invalid punctured length')
    return L


def _qpp_params(K: int) -> tuple[int, int]:
    """Return valid QPP parameters ``(f1, f2)`` for block length ``K``.

    Generic construction ``f1 = 1``, ``f2 = lcm(rad(K), 4 if 4 | K else 1)``
    satisfies the quadratic-permutation conditions (Sun–Takeshita):
    ``gcd(f1, K) = 1``, every prime divisor of ``K`` divides ``f2``, and
    ``4 | f2`` whenever ``4 | K``.  It is therefore bijective for every
    ``K``, including the CCSDS block lengths 1784/3568/7136/8920/16384.
    """
    rad = 1
    m = K
    d = 2
    while d * d <= m:
        if m % d == 0:
            rad *= d
            while m % d == 0:
                m //= d
        d += 1
    if m > 1:
        rad *= m
    f2 = math.lcm(rad, 4) if K % 4 == 0 else rad
    return 1, f2


def _qpp_perm(K: int) -> list[int]:
    """Permutation indices ``π(i) = (f1·i + f2·i²) mod K`` for i = 0..K-1."""
    if K <= 0:
        return []
    f1, f2 = _qpp_params(K)
    return [(f1 * i + f2 * i * i) % K for i in range(K)]


def ccsds_interleaver(bits: list[int]) -> list[int]:
    """Apply the CCSDS quadratic-permutation interleaver.

    Output position ``π(i)`` receives input bit ``i`` (``out[π(i)] = bits[i]``,
    docs/CCSDS_Turbo_Spec.md §2.2).  The mapping is a permutation, so
    ``ccsds_deinterleaver(ccsds_interleaver(bits)) == bits``.  It is **not**
    self-inverse.
    """
    K = len(bits)
    if K < 2:
        return bits[:]
    out: list[int] = [0] * K
    f1, f2 = _qpp_params(K)
    for i in range(K):
        out[(f1 * i + f2 * i * i) % K] = bits[i]
    return out


def ccsds_deinterleaver(bits: list[int]) -> list[int]:
    """Inverse of :func:`ccsds_interleaver` (``out[i] = bits[π(i)]``)."""
    K = len(bits)
    if K < 2:
        return bits[:]
    out: list[int] = [0] * K
    f1, f2 = _qpp_params(K)
    for i in range(K):
        out[i] = bits[(f1 * i + f2 * i * i) % K]
    return out


def _rsc_parity(bits: list[int], flush: int = 0) -> tuple[list[int], list[int]]:
    """Parity bits of one feed-forward constituent encoder (g1 = 0x1B).

    Returns ``(parity, flush_parity)``: one parity bit per input bit with the
    register updated lsb-current (``state = (state << 1 | u) & MASK``),
    followed by ``flush`` parity bits computed while shifting in zeros.  Four
    zero shifts reset the 4-bit register to state 0 (CCSDS §3.2.3
    termination), which the decoder relies on for its backward recursion.
    """
    state = 0
    par: list[int] = []
    for u in bits:
        state = ((state << 1) | u) & MASK
        par.append(_parity(state & GEN))
    flush_par: list[int] = []
    for _ in range(flush):
        state = ((state << 1) | 0) & MASK
        flush_par.append(_parity(state & GEN))
    return par, flush_par


def _rsc_parities(
    bits: list[int], gens: tuple[int, ...], flush: int = 0
) -> tuple[list[list[int]], list[list[int]]]:
    """Parity bits of a feed-forward constituent encoder per generator.

    Like :func:`_rsc_parity` but computes one parity stream per polynomial in
    ``gens`` (used for the rate-1/6 constituent codes, which produce several
    parity outputs).  Returns ``(parities, flush_parities)``, each a list with
    one entry per generator.
    """
    state = 0
    pars: list[list[int]] = [[] for _ in gens]
    for u in bits:
        state = ((state << 1) | u) & MASK
        for j, g in enumerate(gens):
            pars[j].append(_parity(state & g))
    flush_pars: list[list[int]] = [[] for _ in gens]
    for _ in range(flush):
        state = ((state << 1) | 0) & MASK
        for j, g in enumerate(gens):
            flush_pars[j].append(_parity(state & g))
    return pars, flush_pars


def encode(bits: list[int], puncture: bool = False, rate: str | None = None) -> list[int]:
    """Encode *bits* with the CCSDS Turbo scheme.

    Backwards‑compatible signature:
    * ``puncture`` – legacy flag; ``True`` yields the CCSDS punctured
      (rate‑1/4) format, ``False`` yields the full rate‑1/3 stream.
    * ``rate`` – optional explicit rate string (``"1/3"``, ``"1/4"``,
      ``"1/2"`` or ``"1/6"``).  If provided it overrides ``puncture``.

    Supported rates:
    * ``"1/3"`` – full (systematic + parity1 + parity2).
    * ``"1/4"`` – CCSDS punctured pattern (systematic + parity1 + even-indexed
      parity2 + 4 flush bits).
    * ``"1/2"`` – systematic + parity1 + 4 flush bits (parity2 omitted).
    * ``"1/6"`` – CCSDS full rate‑1/6: constituent A (natural order) outputs
      ``G1, G2, G3`` parities, constituent B (interleaved) outputs ``G1, G3``
      parities (CCSDS 131.0-B-4 §3.3.1), plus the flush parity bits of both
      constituents.  Stream layout ``x + pA1 + pA2 + pA3 + pB1 + pB3``
      followed by ``4 + 4 + 4 + 4 + 4`` flush parity bits.

    Args:
        bits: Input payload; each element must be 0 or 1.
        puncture: Legacy flag for rate‑1/4 (default ``False``).
        rate: Optional explicit rate; overrides ``puncture`` when given.

    Returns:
        Encoded bit list.
    """
    if not bits:
        return []
    # Validate bits are 0 or 1
    for i, b in enumerate(bits):
        if b not in (0, 1):
            raise ValueError(f"Bit at position {i} is not 0 or 1: {b}")

    # Determine the effective rate
    if rate is None:
        effective_rate = "1/4" if puncture else "1/3"
    else:
        effective_rate = rate

    if effective_rate == "1/6":
        pA, fA = _rsc_parities(bits, (GEN, GEN2, GEN3), TAIL)
        pB, fB = _rsc_parities(ccsds_interleaver(bits), (GEN, GEN3), TAIL)
        stream = bits + pA[0] + pA[1] + pA[2] + pB[0] + pB[1]
        stream += fA[0] + fA[1] + fA[2] + fB[0] + fB[1]
        return stream

    p1, p1_flush = _rsc_parity(bits, TAIL)
    p2, _ = _rsc_parity(ccsds_interleaver(bits), TAIL)

    if effective_rate == "1/3":
        # Full rate‑1/3 stream
        return bits + p1 + p2
    elif effective_rate == "1/4":
        # CCSDS punctured pattern (even-indexed parity2 + tail)
        return bits + p1 + p2[0::2] + p1_flush
    elif effective_rate == "1/2":
        # Systematic + parity1 + tail (parity2 omitted)
        return bits + p1 + p1_flush
    else:
        raise ValueError(f"Unsupported Turbo code rate: {effective_rate}")


def _puncture(full_bits: list[int]) -> list[int]:
    """Apply the CCSDS puncturing pattern (Rate 1/4) to a 3·L stream.

    ``full_bits`` is ``systematic + parity1 + parity2`` (3 × payload_len).
    The output concatenates the systematic block, the parity1 block, and a
    filtered parity2 block that contains only the bits for *even-indexed*
    payload positions (docs/CCSDS_Turbo_Spec.md §2.3).
    """
    L = len(full_bits) // 3
    systematic = full_bits[:L]
    parity1 = full_bits[L:2 * L]
    parity2 = full_bits[2 * L:]
    # parity2 kept only for even indices
    parity2_filtered = [parity2[i] for i in range(L) if i % 2 == 0]
    return systematic + parity1 + parity2_filtered


def _depuncture(punctured: list[int]) -> list[int]:
    """Re-construct the unpunctured stream from a CCSDS punctured stream.

    Missing ``parity2`` bits (those for odd indices) are filled with ``0`` —
    these positions will be treated as erasures (LLR = 0) by the MAP decoder.
    The function returns ``[systematic, parity1, parity2]`` concatenated.
    """
    L = payload_len_from_punctured(len(punctured))
    systematic = punctured[:L]
    parity1 = punctured[L:2 * L]
    filtered = punctured[2 * L:]
    parity2: list[int] = []
    f_idx = 0
    for i in range(L):
        if i % 2 == 0:
            parity2.append(filtered[f_idx])
            f_idx += 1
        else:
            parity2.append(0)
    return systematic + parity1 + parity2


# --- Log-MAP (BCJR) kernel (numba JIT, per AGENTS.md §4.2) ---


@njit(fastmath=True, cache=True)
def _bcjr_multi_kernel(sys_llr, par_llrs, gens, data_len):
    """Log‑MAP (BCJR) kernel for one constituent code (numba JIT).

    Generalization of :func:`_bcjr_kernel` to several parity outputs:
    ``par_llrs`` is a 2-D float64 array of shape ``(n_par, total)`` and
    ``gens`` the matching generator polynomials; the branch metric sums the
    parity contributions of all outputs.

    ``sys_llr`` is a float64 array of length ``data_len + TAIL``.  For the
    termination steps ``i >= data_len`` only the ``u = 0`` transition is
    allowed, the systematic contribution is zero, and the parity LLR comes
    from ``par_llrs`` (0.0 = erasure when the flush parity was not
    transmitted).  The backward recursion is initialized with
    ``beta[total]`` one-hot at state 0, matching the encoder's flush-to-zero
    termination.

    Args:
        sys_llr: Systematic LLRs (payload positions), zero-padded to
            ``data_len + TAIL`` entries.
        par_llrs: Parity LLRs, shape ``(n_par, data_len + TAIL)``.
        gens: Generator polynomials, length ``n_par``, in lsb-current
            representation.
        data_len: Number of payload positions.

    Returns:
        Posterior LLRs for the ``data_len`` payload positions (positive =
        likelihood of bit 0, per AGENTS.md §2).
    """
    total = sys_llr.shape[0]
    n_states = 16
    neg_inf = -np.inf
    alpha = np.full((total + 1, n_states), neg_inf, dtype=np.float64)
    beta = np.full((total + 1, n_states), neg_inf, dtype=np.float64)
    alpha[0, 0] = 0.0
    beta[total, 0] = 0.0

    # Forward (alpha) recursion
    for i in range(total):
        if i < data_len:
            sl = sys_llr[i]
            n_u = 2
        else:
            sl = 0.0
            n_u = 1  # termination: input bit forced to 0
        for s in range(n_states):
            a = alpha[i, s]
            if a == neg_inf:
                continue
            for u in range(n_u):
                ns = ((s << 1) | u) & MASK
                pbm = 0.0
                for j in range(par_llrs.shape[0]):
                    pj = _parity(ns & gens[j])
                    pbm += par_llrs[j, i] * (1 - 2 * pj)
                bm = (sl * (1 - 2 * u) + pbm) * 0.5
                alpha[i + 1, ns] = np.logaddexp(alpha[i + 1, ns], a + bm)

    # Backward (beta) recursion
    for i in range(total - 1, -1, -1):
        if i < data_len:
            sl = sys_llr[i]
            n_u = 2
        else:
            sl = 0.0
            n_u = 1
        for s in range(n_states):
            for u in range(n_u):
                ns = ((s << 1) | u) & MASK
                b = beta[i + 1, ns]
                if b == neg_inf:
                    continue
                pbm = 0.0
                for j in range(par_llrs.shape[0]):
                    pj = _parity(ns & gens[j])
                    pbm += par_llrs[j, i] * (1 - 2 * pj)
                bm = (sl * (1 - 2 * u) + pbm) * 0.5
                beta[i, s] = np.logaddexp(beta[i, s], b + bm)

    # Posterior LLR for each payload position
    post = np.empty(data_len, dtype=np.float64)
    for i in range(data_len):
        sl = sys_llr[i]
        L0 = neg_inf
        L1 = neg_inf
        for s in range(n_states):
            a = alpha[i, s]
            if a == neg_inf:
                continue
            for u in (0, 1):
                ns = ((s << 1) | u) & MASK
                b = beta[i + 1, ns]
                if b == neg_inf:
                    continue
                pbm = 0.0
                for j in range(par_llrs.shape[0]):
                    pj = _parity(ns & gens[j])
                    pbm += par_llrs[j, i] * (1 - 2 * pj)
                bm = (sl * (1 - 2 * u) + pbm) * 0.5
                prob = a + bm + b
                if u == 0:
                    L0 = np.logaddexp(L0, prob)
                else:
                    L1 = np.logaddexp(L1, prob)
        post[i] = L0 - L1
    return post


@njit(fastmath=True, cache=True)
def _bcjr_kernel(sys_llr, par_llr, gen, data_len):
    """Log‑MAP (BCJR) kernel for one constituent code (numba JIT).

    ``sys_llr`` / ``par_llr`` are float64 arrays of length
    ``data_len + TAIL``.  For the termination steps ``i >= data_len`` only the
    ``u = 0`` transition is allowed, the systematic contribution is zero, and
    the parity LLR comes from ``par_llr`` (0.0 = erasure when the flush parity
    was not transmitted).  The backward recursion is initialized with
    ``beta[total]`` one-hot at state 0, matching the encoder's flush-to-zero
    termination.

    Args:
        sys_llr: Systematic LLRs (payload positions), zero-padded to
            ``data_len + TAIL`` entries.
        par_llr: Parity LLRs, ``data_len + TAIL`` entries (flush steps may be
            real parity or erasures).
        gen: Generator polynomial (0x1B) in lsb-current representation.
        data_len: Number of payload positions.

    Returns:
        Posterior LLRs for the ``data_len`` payload positions (positive =
        likelihood of bit 0, per AGENTS.md §2).
    """
    par_llrs = par_llr.reshape(1, par_llr.shape[0])
    gens = np.empty(1, dtype=np.int64)
    gens[0] = gen
    return _bcjr_multi_kernel(sys_llr, par_llrs, gens, data_len)


# JIT-compile the BCJR kernels at import time so that a timed first decode()
# call (e.g. the performance benchmark) does not pay the compilation cost.
_bcjr_kernel(np.zeros(TAIL + 1, dtype=np.float64), np.zeros(TAIL + 1, dtype=np.float64), GEN, 1)
_bcjr_multi_kernel(
    np.zeros(TAIL + 1, dtype=np.float64),
    np.zeros((3, TAIL + 1), dtype=np.float64),
    np.array([GEN, GEN2, GEN3], dtype=np.int64),
    1,
)


def _llr_array(bits_with_erasures: list[int]) -> np.ndarray:
    """Map 0/1 bits to LLRs (+LLR_0 / LLR_1); ``-1`` marks an erasure → 0.0."""
    a = np.asarray(bits_with_erasures, dtype=np.int64)
    return np.where(a == 0, LLR_0, np.where(a == 1, LLR_1, 0.0)).astype(np.float64)


def _decode_core(
    sys_bits: list[int],
    p1_bits: list[int],
    p2_bits: list[int],
    iterations: int,
    p1_tail_bits: list[int] | None = None,
) -> list[int]:
    """Iterative Log‑MAP decoding of one rate‑1/3 Turbo frame.

    Runs ``iterations`` turbo iterations (docs/CCSDS_Turbo_Spec.md §2.5):
    BCJR on the first constituent with the systematic+apriori LLRs, the
    extrinsic is interleaved and fed with the interleaved systematic LLRs into
    the second constituent's BCJR, and the de-interleaved extrinsic becomes
    the next iteration's apriori.  The final hard decision uses
    ``sys + apriori`` (LLR ≥ 0 → bit 0).

    With ``iterations == 0`` the MAP loop is skipped and the systematic bits
    are returned directly.

    Args:
        sys_bits: Systematic bits (length L).
        p1_bits: Parity bits of the first constituent (length L).
        p2_bits: Parity bits of the second constituent (length L; odd
            positions may be ``-1`` erasure markers in punctured mode).
        iterations: Number of turbo iterations (default 5).
        p1_tail_bits: The 4 flush parity bits of the first constituent
            (punctured frames), or ``None`` to treat the flush parity as
            erasures (full rate‑1/3 frames, which carry no tail).

    Returns:
        The recovered payload bits.
    """
    L = len(sys_bits)
    sys_llr = _llr_array(sys_bits)
    p1_llr = _llr_array(p1_bits + list(p1_tail_bits if p1_tail_bits is not None else [-1] * TAIL))
    p2_llr = _llr_array(p2_bits + [-1] * TAIL)
    if iterations <= 0:
        return [0 if v >= 0 else 1 for v in sys_llr]

    perm = np.asarray(_qpp_perm(L), dtype=np.int64)
    apriori = np.zeros(L, dtype=np.float64)
    for _ in range(iterations):
        # 1st constituent: systematic + apriori, parity p1
        sys1 = sys_llr + apriori
        post1 = _bcjr_kernel(
            np.concatenate((sys1, np.zeros(TAIL, dtype=np.float64))),
            p1_llr,
            GEN,
            L,
        )
        ext1 = post1 - sys1
        # interleave: out[π(i)] = in[i]
        iext1 = np.empty(L, dtype=np.float64)
        iext1[perm] = ext1
        inter_sys = np.empty(L, dtype=np.float64)
        inter_sys[perm] = sys_llr

        # 2nd constituent: interleaved systematic + interleaved extrinsic, parity p2
        sys2 = inter_sys + iext1
        post2 = _bcjr_kernel(
            np.concatenate((sys2, np.zeros(TAIL, dtype=np.float64))),
            p2_llr,
            GEN,
            L,
        )
        ext2 = post2 - sys2
        # de-interleave: out[i] = in[π(i)]
        apriori = ext2[perm]

    final = sys_llr + apriori
    return [0 if v >= 0 else 1 for v in final]


def _decode_core_rate16(
    sys_bits: list[int],
    a1_bits: list[int],
    a2_bits: list[int],
    a3_bits: list[int],
    b1_bits: list[int],
    b3_bits: list[int],
    iterations: int,
) -> list[int]:
    """Iterative Log‑MAP decoding of one rate‑1/6 Turbo frame.

    Constituent A (natural order) contributes the ``G1, G2, G3`` parity
    streams ``a1/a2/a3``, constituent B (interleaved order) the ``G1, G3``
    streams ``b1/b3`` (CCSDS 131.0-B-4 §3.3.1).  Each list already carries its
    4 flush parity bits at the end, which terminate the trellis at state 0.

    Args:
        sys_bits: Systematic bits (length L).
        a1_bits/a2_bits/a3_bits: Parity streams of constituent A
            (length L + TAIL).
        b1_bits/b3_bits: Parity streams of constituent B (length L + TAIL).
        iterations: Number of turbo iterations.

    Returns:
        The recovered payload bits.
    """
    L = len(sys_bits)
    sys_llr = _llr_array(sys_bits)
    a_par = np.stack(
        [_llr_array(a1_bits), _llr_array(a2_bits), _llr_array(a3_bits)]
    )
    b_par = np.stack([_llr_array(b1_bits), _llr_array(b3_bits)])
    gens_a = np.array([GEN, GEN2, GEN3], dtype=np.int64)
    gens_b = np.array([GEN, GEN3], dtype=np.int64)
    if iterations <= 0:
        return [0 if v >= 0 else 1 for v in sys_llr]

    perm = np.asarray(_qpp_perm(L), dtype=np.int64)
    apriori = np.zeros(L, dtype=np.float64)
    tail_zeros = np.zeros(TAIL, dtype=np.float64)
    for _ in range(iterations):
        # 1st constituent (natural order): systematic + apriori, 3 parities
        sys1 = sys_llr + apriori
        post1 = _bcjr_multi_kernel(
            np.concatenate((sys1, tail_zeros)),
            a_par,
            gens_a,
            L,
        )
        ext1 = post1 - sys1
        # interleave: out[π(i)] = in[i]
        iext1 = np.empty(L, dtype=np.float64)
        iext1[perm] = ext1
        inter_sys = np.empty(L, dtype=np.float64)
        inter_sys[perm] = sys_llr

        # 2nd constituent (interleaved order): systematic + extrinsic, 2 parities
        sys2 = inter_sys + iext1
        post2 = _bcjr_multi_kernel(
            np.concatenate((sys2, tail_zeros)),
            b_par,
            gens_b,
            L,
        )
        ext2 = post2 - sys2
        # de-interleave: out[i] = in[π(i)]
        apriori = ext2[perm]

    final = sys_llr + apriori
    return [0 if v >= 0 else 1 for v in final]


def decode(
    punctured_bits: list[int], iterations: int = 5, rate: str | None = None
) -> list[int]:
    """Decode a Turbo stream (punctured rate‑1/4, unpunctured rate‑1/3, or rate‑1/6).

    The stream format is detected from its length unless ``rate`` is given:
    a punctured frame ends with the 4 flush bits, so if
    ``payload_len_from_punctured(len - 4)`` yields a valid payload length the
    input is decoded as punctured; otherwise a length divisible by 3 is
    decoded as a full rate‑1/3 frame.  Rate‑1/6 streams are **not**
    auto‑detected (their length can coincide with punctured rate‑1/4 frames);
    pass ``rate="1/6"`` explicitly.

    Args:
        punctured_bits: Encoded bit stream (each element 0 or 1).
        iterations: Number of turbo iterations (default 5); 0 skips the MAP
            loop and returns the systematic bits.
        rate: Optional explicit rate string (``"1/6"``); bypasses
            length-based detection.

    Returns:
        The recovered payload bits.
    """
    stream = list(punctured_bits)
    if not stream:
        return []
    if rate == "1/6":
        # Rate‑1/6 layout: x + pA1 + pA2 + pA3 + pB1 + pB3
        # followed by flush parity bits (4 each: fA1, fA2, fA3, fB1, fB3).
        total = len(stream)
        if total < 20 or (total - 20) % 6 != 0:
            raise ValueError(
                f"Invalid rate-1/6 stream length {total}: expected 6*L + 20"
            )
        L = (total - 20) // 6
        x = stream[:L]
        pA1 = stream[L:2 * L]
        pA2 = stream[2 * L:3 * L]
        pA3 = stream[3 * L:4 * L]
        pB1 = stream[4 * L:5 * L]
        pB3 = stream[5 * L:6 * L]
        fA1 = stream[6 * L : 6 * L + TAIL]
        fA2 = stream[6 * L + TAIL : 6 * L + 2 * TAIL]
        fA3 = stream[6 * L + 2 * TAIL : 6 * L + 3 * TAIL]
        fB1 = stream[6 * L + 3 * TAIL : 6 * L + 4 * TAIL]
        fB3 = stream[6 * L + 4 * TAIL : 6 * L + 5 * TAIL]
        return _decode_core_rate16(
            x,
            pA1 + fA1,
            pA2 + fA2,
            pA3 + fA3,
            pB1 + fB1,
            pB3 + fB3,
            iterations,
        )
    # Try punctured (rate‑1/4) detection first
    try:
        L = payload_len_from_punctured(len(stream) - TAIL)
    except ValueError:
        L = None
    if L is not None:
        # punctured frame: systematic + parity1 + even parity2 + 4 flush bits
        sys = stream[:L]
        p1 = stream[L:2 * L]
        p2_filt = stream[2 * L : 2 * L + (L + 1) // 2]
        tail = stream[2 * L + (L + 1) // 2 : 2 * L + (L + 1) // 2 + TAIL]
        # odd parity2 positions were punctured → erasures (-1 markers)
        p2: list[int] = []
        f_idx = 0
        for i in range(L):
            if i % 2 == 0:
                p2.append(p2_filt[f_idx])
                f_idx += 1
            else:
                p2.append(-1)
        return _decode_core(sys, p1, p2, iterations, p1_tail_bits=tail)
    # Try half‑rate (1/2) detection: systematic + parity1 + 4‑flush tail
    if (len(stream) - TAIL) % 2 == 0:
        L_half = (len(stream) - TAIL) // 2
        sys = stream[:L_half]
        p1 = stream[L_half:2 * L_half]
        tail = stream[2 * L_half:2 * L_half + TAIL]
        # Verify parity (fallback – no error correction); on mismatch fall
        # through to the rate‑1/3 interpretation instead of raising, since a
        # full rate‑1/3 frame of even length also satisfies the length check.
        expected = _rsc_parity(sys, TAIL)[0] + tail
        if expected == p1 + tail:
            return sys
    # If not punctured nor half‑rate, fall back to full rate‑1/3
    if len(stream) % 3 != 0:
        raise ValueError(
            "Unrecognized Turbo stream: neither punctured (rate-1/4), half‑rate (1/2) nor full (rate-1/3)"
        )
    # unpunctured frame: systematic + parity1 + parity2 (no tail transmitted)
    L = len(stream) // 3
    return _decode_core(stream[:L], stream[L:2 * L], stream[2 * L:], iterations)


def decode_unpunctured(turbo_bits: list[int], iterations: int = 3) -> list[int]:
    """Decode a full rate‑1/3 (unpunctured) Turbo stream.

    Args:
        turbo_bits: Encoded bit stream ``systematic + parity1 + parity2``
            (length a multiple of 3).
        iterations: Number of turbo iterations (default 3).

    Returns:
        The recovered payload bits.
    """
    if len(turbo_bits) % 3 != 0:
        raise ValueError("Unpunctured Turbo stream length must be a multiple of 3")
    N = len(turbo_bits) // 3
    return _decode_core(
        list(turbo_bits)[:N],
        list(turbo_bits)[N:2 * N],
        list(turbo_bits)[2 * N:],
        iterations,
    )


def _decode_padded_rate16(bits: list[int]) -> list[int]:
    """Decode a byte-padded rate‑1/6 stream (CLI helper).

    A rate‑1/6 frame has length ``6*L + 20``, which for byte‑aligned payloads
    (``L ≡ 0 mod 8``) is always congruent to 4 mod 8, so the CLI packs 4
    trailing padding bits.  The true length is the unique candidate ``T`` in
    ``[len-7, len]`` satisfying ``T ≡ 20 (mod 6)``.
    """
    n = len(bits)
    for k in range(8):
        t = n - k
        if t >= 20 and (t - 20) % 6 == 0:
            return decode(bits[:t], rate="1/6")
    raise ValueError(
        "Unrecognized rate-1/6 stream: no valid length within the last 8 bits"
    )


def main_encode(rate: str | None = None) -> None:
    import sys

    data = sys.stdin.buffer.read()
    from .utils import bits_to_bytes, bytes_to_bits

    bits = bytes_to_bits(data)
    enc = encode(bits, rate=rate)
    sys.stdout.buffer.write(bits_to_bytes(enc))


def main_decode(rate: str | None = None) -> None:
    import sys

    data = sys.stdin.buffer.read()
    from .utils import bits_to_bytes, bytes_to_bits

    bits = bytes_to_bits(data)
    # Detect the stream format: full rate‑1/3 streams have length divisible
    # by 3; punctured rate‑1/4 streams (with the 4 flush bits) do not in
    # general.  Rate‑1/6 frames require the explicit ``--rate 1/6`` flag and
    # are decoded with padding-trimming, since 6L+20 is never byte-aligned.
    if rate == "1/6":
        dec = _decode_padded_rate16(bits)
    elif rate is not None:
        dec = decode(bits, rate=rate)
    elif len(bits) % 3 == 0:
        dec = decode_unpunctured(bits)
    else:
        dec = decode(bits)
    sys.stdout.buffer.write(bits_to_bytes(dec))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CCSDS Turbo coder")
    parser.add_argument("mode", choices=["encode", "decode"], help="operation mode")
    args = parser.parse_args()
    if args.mode == "encode":
        main_encode()
    else:
        main_decode()
