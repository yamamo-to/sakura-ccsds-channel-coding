"""CCSDS Turbo encoder/decoder (rates 1/2, 1/3, 1/4, 1/6; iterative Log-MAP).

Implements the Turbo code of CCSDS 131.0-B-4 §3 / §5 / §6 with the
**recursive** (feedback) RSC constituent codes of §3.3.1 and the
**block interleaver** of §6.3g:

* **Constituent code** – constraint length ``K = 5`` (4-bit shift register),
  feedback polynomial ``g0 = 10011_2 (23_8)`` (feedback ``fb = D3 ^ D4``,
  newest register bit ``D1``) and forward polynomials over
  ``[new_D1, D1, D2, D3, D4]`` (bits 4..0 of the generator integer):
  ``G1 = 11011_2 (33_8)``, ``G2 = 10101_2 (25_8)``, ``G3 = 11111_2 (37_8)``.
  The systematic output is ``G_SYS = 10011_2 (23_8)``.  The register is
  flushed to zero over the last 4 steps by feeding the feedback bit back
  (CCSDS §3.2.3 termination), giving stream length ``K + 4`` per output.
* **Interleaver** – CCSDS §6.3g quadratic-permutation block interleaver
  (:mod:`ccsds_codec.core.interleaver`, ``interleaved[j] = bits[perm[j]]``).
* **Rates / stream layout** (transmitted order, block-per-codeword):
  * rate 1/2 – ``2*(K+4)`` bits: ``[s_c, p_c]`` where ``p_c`` is the upper
    ``G1`` parity for even codewords and the lower ``G1`` parity for odd
    codewords (CCSDS §3.4.2 puncturing of the rate-1/3 code);
  * rate 1/3 – ``3*(K+4)`` bits: ``[s_c, uG1_c, lG1_c]``;
  * rate 1/4 – ``4*(K+4)`` bits: ``[s_c, uG2_c, uG3_c, lG1_c]``;
  * rate 1/6 – ``6*(K+4)`` bits: ``[s_c, uG1_c, uG2_c, uG3_c, lG1_c, lG3_c]``.
  Upper constituent gens: 1/2,1/3 → ``[SYS,G1]``; 1/4 → ``[SYS,G2,G3]``;
  1/6 → ``[SYS,G1,G2,G3]``.  Lower constituent gens: 1/2,1/3,1/4 → ``[G1]``;
  1/6 → ``[G1,G3]``.
* **Decoder** – iterative Log-MAP (BCJR) with true extrinsic separation:
  ``γ(s',u) = 0.5·Σ_c ch_llr[c]·x_c ± La/2``,
  ``LLR(u_k) = max*_{u=0}(α+γ+β) − max*_{u=1}(α+γ+β)``,
  ``ext = LLR − La − ch_sys``.  Numerically stable through
  ``np.logaddexp`` (no exp overflow/underflow).  LLR convention: positive
  value = likelihood of bit 0, negative = likelihood of bit 1 (AGENTS.md §2).
  The final hard decision uses the de-interleaved APP of the lower
  constituent (golden-verified against the gr-ccsds-1 / SatDump reference).

The encoder output is bit-exact against the CCSDS golden vectors
(K = 1784, 3568; all four rates; ``tests/test_turbo_golden.py``).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .convolutional import _parity
from .interleaver import ccsds_deinterleaver, ccsds_interleaver, ccsds_perm

__all__ = [
    "GEN",
    "GEN_SYS",
    "GEN2",
    "GEN3",
    "LLR_0",
    "LLR_1",
    "TAIL",
    "ccsds_deinterleaver",
    "ccsds_interleaver",
    "decode",
    "decode_padded_rate16",
    "decode_unpunctured",
    "encode",
]

# Constituent-code generator polynomials (CCSDS 131.0-B-4 §3.3.1) as integers
# whose bits 4..0 are the coefficients of [new_D1, D1, D2, D3, D4]:
#   G_SYS = 10011_2 -> output = new_D1 ^ D3 ^ D4 (= the info bit u)
#   G1    = 11011_2 -> output = new_D1 ^ D1 ^ D4
#   G2    = 10101_2 -> output = new_D1 ^ D2 ^ D4
#   G3    = 11111_2 -> output = new_D1 ^ D1 ^ D2 ^ D3 ^ D4
GEN_SYS = 0x13
GEN = 0x1B  # G1 (kept name for backwards compatibility)
GEN2 = 0x15  # G2
GEN3 = 0x1F  # G3

#: Termination (flush) steps per constituent code (K - 1 = 4 for K = 5).
TAIL = 4

# LLR scale (AGENTS.md §2): bit 0 -> +1.0, bit 1 -> -1.0
LLR_0 = 1.0
LLR_1 = -1.0

#: Codeword components per rate (stream length = NCOMP[rate] * (K + TAIL)).
NCOMP = {"1/2": 2, "1/3": 3, "1/4": 4, "1/6": 6}

#: Constituent generator lists per rate (index 0 is the systematic output).
_UPPER_GENS = {
    "1/2": [GEN_SYS, GEN],
    "1/3": [GEN_SYS, GEN],
    "1/4": [GEN_SYS, GEN2, GEN3],
    "1/6": [GEN_SYS, GEN, GEN2, GEN3],
}
_LOWER_GENS = {
    "1/2": [GEN_SYS, GEN],
    "1/3": [GEN_SYS, GEN],
    "1/4": [GEN_SYS, GEN],
    "1/6": [GEN_SYS, GEN, GEN3],
}

#: CCSDS block lengths (CCSDS 131.0-B-4 §3.1.1).
STANDARD_K = (1784, 3568, 7136, 8920, 16384)


def _rsc_streams(bits: list[int], gens: list[int]) -> list[list[int]]:
    """Encode one recursive RSC constituent, one stream per generator.

    Implements the recursive shift register of CCSDS 131.0-B-4 §3.3.1:
    ``fb = D3 ^ D4``, ``new_D1 = u ^ fb``, ``ns = (state >> 1) | (new_D1 << 3)``.
    The last ``TAIL`` steps terminate the register at state 0 by feeding the
    feedback bit back as input (CCSDS §3.2.3).

    Args:
        bits: Payload bits (each 0 or 1).
        gens: Generator polynomials (int, bits 4..0 = [new_D1..D4]).

    Returns:
        One list of length ``len(bits) + TAIL`` per generator.
    """
    state = 0
    streams: list[list[int]] = [[] for _ in gens]
    seq: list[int | None] = list(bits) + [None] * TAIL
    for u in seq:
        D1, D2, D3, D4 = (state >> 3) & 1, (state >> 2) & 1, (state >> 1) & 1, state & 1
        fb = D3 ^ D4
        if u is None:
            u = fb  # termination: input = feedback -> register flushes to 0
        new_D1 = u ^ fb
        ns = (state >> 1) | (new_D1 << 3)
        for g, st in zip(gens, streams):
            v = (
                ((g >> 4) & 1) * new_D1
                ^ ((g >> 3) & 1) * D1
                ^ ((g >> 2) & 1) * D2
                ^ ((g >> 1) & 1) * D3
                ^ (g & 1) * D4
            )
            st.append(_parity(v))
        state = ns
    return streams


def encode(bits: list[int], puncture: bool = False, rate: str | None = None) -> list[int]:
    """Encode *bits* with the CCSDS Turbo scheme (recursive RSC + §6.3g).

    Backwards-compatible signature:
    * ``rate`` – explicit rate string (``"1/2"``, ``"1/3"``, ``"1/4"`` or
      ``"1/6"``); overrides ``puncture`` when given.
    * ``puncture`` – legacy flag: ``True`` selects the punctured rate‑1/2
      code, ``False`` the full rate‑1/3 code.

    Stream layouts (CCSDS 131.0-B-4 §3.4 / golden-verified):
    * ``"1/2"`` – ``2*(K+4)``: ``[s_c, par_c]`` with ``par_c`` = upper G1 at
      even codewords, lower G1 at odd codewords;
    * ``"1/3"`` – ``3*(K+4)``: ``[s_c, uG1_c, lG1_c]``;
    * ``"1/4"`` – ``4*(K+4)``: ``[s_c, uG2_c, uG3_c, lG1_c]``;
    * ``"1/6"`` – ``6*(K+4)``: ``[s_c, uG1_c, uG2_c, uG3_c, lG1_c, lG3_c]``.

    Args:
        bits: Input payload; each element must be 0 or 1.
        puncture: Legacy flag for rate 1/2 (default ``False``).
        rate: Optional explicit rate; overrides ``puncture`` when given.

    Returns:
        Encoded bit list of length ``NCOMP[rate] * (K + TAIL)``.
    """
    if not bits:
        return []
    for i, b in enumerate(bits):
        if b not in (0, 1):
            raise ValueError(f"Bit at position {i} is not 0 or 1: {b}")

    effective_rate = rate if rate is not None else ("1/2" if puncture else "1/3")
    if effective_rate not in NCOMP:
        raise ValueError(f"Unsupported Turbo code rate: {effective_rate}")

    K = len(bits)
    ibits = ccsds_interleaver(bits)
    ncomp = NCOMP[effective_rate]

    if effective_rate in ("1/2", "1/3"):
        upper = _rsc_streams(bits, _UPPER_GENS[effective_rate])  # [sys, G1]
        lower = _rsc_streams(ibits, _LOWER_GENS[effective_rate])  # [sys, G1]
        blocks = [
            [upper[0][i], upper[1][i], lower[1][i]] for i in range(K + TAIL)
        ]
        if effective_rate == "1/2":
            # Puncture: even codeword keeps upper G1, odd keeps lower G1.
            out: list[int] = []
            for cw, blk in enumerate(blocks):
                out.append(blk[0])
                out.append(blk[1] if cw % 2 == 0 else blk[2])
            return out
        return [b for blk in blocks for b in blk]

    if effective_rate == "1/4":
        upper = _rsc_streams(bits, _UPPER_GENS["1/4"])  # [sys, G2, G3]
        lower = _rsc_streams(ibits, _LOWER_GENS["1/4"])  # [sys, G1]
        blocks = [
            [upper[0][i], upper[1][i], upper[2][i], lower[1][i]]
            for i in range(K + TAIL)
        ]
        return [b for blk in blocks for b in blk]

    # rate 1/6
    upper = _rsc_streams(bits, _UPPER_GENS["1/6"])  # [sys, G1, G2, G3]
    lower = _rsc_streams(ibits, _LOWER_GENS["1/6"])  # [sys, G1, G3]
    blocks = [
        [upper[0][i], upper[1][i], upper[2][i], upper[3][i], lower[1][i], lower[2][i]]
        for i in range(K + TAIL)
    ]
    return [b for blk in blocks for b in blk]


# --- trellis tables ---------------------------------------------------------


def _build_trellis(gens: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the recursive-RSC trellis tables for one constituent code.

    Returns ``(ns, x, pred0, pred1)`` with (state bits: bit3 = D1 newest …
    bit0 = D4 oldest):

    * ``ns[s, u]`` – next state for input bit ``u``;
    * ``x[s, u, c]`` – BPSK channel symbol (bit 0 -> +1, bit 1 -> -1) of
      component ``c`` (component 0 is the systematic output);
    * ``pred0[s]`` / ``pred1[s]`` – the unique predecessor state ``s'`` with
      ``ns[s', 0] == s`` / ``ns[s', 1] == s``.
    """
    ns = np.empty((16, 2), dtype=np.int64)
    x = np.empty((16, 2, len(gens)), dtype=np.float64)
    for s in range(16):
        D1, D2, D3, D4 = (s >> 3) & 1, (s >> 2) & 1, (s >> 1) & 1, s & 1
        fb = D3 ^ D4
        for u in (0, 1):
            new_D1 = u ^ fb
            ns[s, u] = (s >> 1) | (new_D1 << 3)
            for gi, g in enumerate(gens):
                v = (
                    ((g >> 4) & 1) * new_D1
                    ^ ((g >> 3) & 1) * D1
                    ^ ((g >> 2) & 1) * D2
                    ^ ((g >> 1) & 1) * D3
                    ^ (g & 1) * D4
                )
                x[s, u, gi] = 1.0 - 2.0 * _parity(v)
    pred0 = np.empty(16, dtype=np.int64)
    pred1 = np.empty(16, dtype=np.int64)
    for s in range(16):
        pred0[s] = next(s0 for s0 in range(16) if ns[s0, 0] == s)
        pred1[s] = next(s0 for s0 in range(16) if ns[s0, 1] == s)
    return ns, x, pred0, pred1


# --- Log-MAP (BCJR) kernel (numba JIT, per AGENTS.md §4.2) -------------------


@njit(fastmath=True, cache=True)
def _bcjr_kernel(ch, la, ns, x, pred0, pred1, data_len):
    """Log‑MAP (BCJR) kernel for one recursive RSC constituent (numba JIT).

    Golden-verified formulation (see scratch_turbo_decoder.py):
    ``γ(s',u) = 0.5·Σ_c ch[c,i]·x[s',u,c] ± La/2`` with the a-priori term
    applied only to the ``data_len`` information steps; the ``TAIL``
    termination steps keep both input values (the encoder's termination
    input is the feedback bit, not a forced zero) and carry no prior.
    The trellis is terminated: α starts one-hot at state 0 and β ends
    one-hot at state 0 (CCSDS §3.2.3).  Per-step max normalisation keeps
    the recursion numerically bounded.

    Args:
        ch: Channel LLRs, float64 array of shape ``(ncomp, N)`` with
            ``N = data_len + TAIL``; punctured positions must be 0.0.
        la: A-priori LLRs, float64 array of shape ``(data_len,)``.
        ns: Next-state table ``(16, 2)`` int64.
        x: BPSK output table ``(16, 2, ncomp)`` float64 (bit 0 -> +1).
        pred0: Predecessor table for input 0, ``(16,)`` int64.
        pred1: Predecessor table for input 1, ``(16,)`` int64.
        data_len: Number of information (payload) positions.

    Returns:
        Tuple ``(ext, app)`` of float64 arrays of shape ``(data_len,)``:
        the extrinsic LLRs and the full APP LLRs (positive = bit 0).
    """
    ncomp = ch.shape[0]
    N = ch.shape[1]
    n_states = 16
    neg_inf = -np.inf

    # gamma[i, s, u] = 0.5 * sum_c ch[c, i] * x[s, u, c]
    gamma = np.empty((N, n_states, 2), dtype=np.float64)
    for i in range(N):
        for s in range(n_states):
            for u in range(2):
                acc = 0.0
                for c in range(ncomp):
                    acc += x[s, u, c] * ch[c, i]
                gamma[i, s, u] = 0.5 * acc
    for i in range(data_len):
        for s in range(n_states):
            gamma[i, s, 0] += 0.5 * la[i]
            gamma[i, s, 1] -= 0.5 * la[i]

    # Forward (alpha) recursion
    alpha = np.full((N + 1, n_states), neg_inf, dtype=np.float64)
    alpha[0, 0] = 0.0
    for i in range(N):
        g0 = gamma[i, :, 0]
        g1 = gamma[i, :, 1]
        a = np.logaddexp(alpha[i][pred0] + g0[pred0], alpha[i][pred1] + g1[pred1])
        alpha[i + 1] = a - a.max()

    # Backward (beta) recursion
    beta = np.full((N + 1, n_states), neg_inf, dtype=np.float64)
    beta[N, 0] = 0.0
    for i in range(N - 1, -1, -1):
        g0 = gamma[i, :, 0]
        g1 = gamma[i, :, 1]
        b = np.logaddexp(beta[i + 1][ns[:, 0]] + g0, beta[i + 1][ns[:, 1]] + g1)
        beta[i] = b - b.max()

    # Posterior / extrinsic LLRs for the information positions
    ext = np.empty(data_len, dtype=np.float64)
    app = np.empty(data_len, dtype=np.float64)
    for i in range(data_len):
        g0 = gamma[i, :, 0]
        g1 = gamma[i, :, 1]
        t0 = alpha[i] + g0 + beta[i + 1][ns[:, 0]]
        t1 = alpha[i] + g1 + beta[i + 1][ns[:, 1]]
        m0 = t0.max()
        m1 = t1.max()
        l0 = m0 + np.log(np.sum(np.exp(t0 - m0)))
        l1 = m1 + np.log(np.sum(np.exp(t1 - m1)))
        llr = l0 - l1
        app[i] = llr
        ext[i] = llr - la[i] - ch[0, i]
    return ext, app


# JIT-compile the BCJR kernel at import time so that a timed first decode()
# call (e.g. the performance benchmark) does not pay the compilation cost.
_bcjr_kernel(
    np.zeros((2, TAIL + 1), dtype=np.float64),
    np.zeros(1, dtype=np.float64),
    np.zeros((16, 2), dtype=np.int64),
    np.zeros((16, 2, 2), dtype=np.float64),
    np.zeros(16, dtype=np.int64),
    np.zeros(16, dtype=np.int64),
    1,
)


def _llr_array(bits: list[int]) -> np.ndarray:
    """Map 0/1 bits to LLRs (+LLR_0 / LLR_1); any other value is an erasure → 0.0."""
    a = np.asarray(bits, dtype=np.float64)
    return np.where(a == 0.0, LLR_0, np.where(a == 1.0, LLR_1, 0.0)).astype(np.float64)


def _demux(rx: np.ndarray, rate: str, K: int, perm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the received LLR sequence into upper/lower constituent streams.

    ``rx`` has shape ``(NCOMP[rate] * (K + TAIL),)`` (codeword-major layout).
    The lower constituent's systematic channel is the interleaved systematic
    LLRs for the ``K`` information positions, zero-padded over the ``TAIL``
    termination steps (the lower code transmits no systematic there).  For
    rate 1/2 the punctured parity positions become erasures (0.0).

    Args:
        rx: Received channel LLRs (float64, 1-D).
        rate: Code rate (``"1/2"``, ``"1/3"``, ``"1/4"`` or ``"1/6"``).
        K: Payload length in bits.
        perm: Interleaver permutation (int64 array of length ``K``).

    Returns:
        Tuple ``(upper, lower)`` of float64 arrays of shape
        ``(ncomp, K + TAIL)`` with component 0 = systematic.
    """
    N = K + TAIL
    rx2 = rx.reshape(N, NCOMP[rate])
    sys = rx2[:, 0]
    sys_int = np.concatenate([sys[perm], np.zeros(TAIL, dtype=np.float64)])
    cw = np.arange(N)
    if rate == "1/2":
        upper = np.stack([sys, np.where(cw % 2 == 0, rx2[:, 1], 0.0)])
        lower = np.stack([sys_int, np.where(cw % 2 == 1, rx2[:, 1], 0.0)])
    elif rate == "1/3":
        upper = np.stack([sys, rx2[:, 1]])
        lower = np.stack([sys_int, rx2[:, 2]])
    elif rate == "1/4":
        upper = np.stack([sys, rx2[:, 1], rx2[:, 2]])
        lower = np.stack([sys_int, rx2[:, 3]])
    else:  # 1/6
        upper = np.stack([sys, rx2[:, 1], rx2[:, 2], rx2[:, 3]])
        lower = np.stack([sys_int, rx2[:, 4], rx2[:, 5]])
    return upper, lower


def _turbo_decode_core(rx: np.ndarray, rate: str, K: int, iterations: int) -> np.ndarray:
    """Iterative Log‑MAP decoding of one Turbo frame from channel LLRs.

    Runs ``iterations`` turbo iterations: BCJR on the upper constituent with
    the systematic+apriori LLRs, the extrinsic is interleaved and fed with
    the interleaved systematic LLRs into the lower constituent's BCJR, and
    the de-interleaved lower extrinsic becomes the next iteration's apriori.
    The final hard decision uses the de-interleaved APP of the lower
    constituent (``LLR ≤ 0 -> bit 1``).

    Args:
        rx: Received channel LLRs of the full transmitted sequence
            (float64, length ``NCOMP[rate] * (K + TAIL)``).
        rate: Code rate (``"1/2"``, ``"1/3"``, ``"1/4"`` or ``"1/6"``).
        K: Payload length in bits.
        iterations: Number of turbo iterations (>= 1).

    Returns:
        The recovered payload bits (uint8 array of length ``K``).
    """
    perm = np.asarray(ccsds_perm(K), dtype=np.int64)
    upper_ch, lower_ch = _demux(rx, rate, K, perm)
    ns_u, x_u, p0u, p1u = _build_trellis(_UPPER_GENS[rate])
    ns_l, x_l, p0l, p1l = _build_trellis(_LOWER_GENS[rate])
    la = np.zeros(K, dtype=np.float64)
    app = np.zeros(K, dtype=np.float64)
    for _ in range(iterations):
        ext_u, _ = _bcjr_kernel(upper_ch, la, ns_u, x_u, p0u, p1u, K)
        la_l = ext_u[perm]
        ext_l, app_l = _bcjr_kernel(lower_ch, la_l, ns_l, x_l, p0l, p1l, K)
        la = np.empty(K, dtype=np.float64)
        la[perm] = ext_l
        app = np.empty(K, dtype=np.float64)
        app[perm] = app_l
    return (app <= 0).astype(np.uint8)


def _k_for_rate(stream_len: int, rate: str) -> int:
    """Return ``K`` for a rate-*rate* stream of length *stream_len*."""
    ncomp = NCOMP[rate]
    if stream_len < ncomp * (TAIL + 1):
        raise ValueError(f"Rate-{rate} stream too short: {stream_len} bits")
    if stream_len % ncomp != 0:
        raise ValueError(
            f"Rate-{rate} stream length {stream_len} is not a multiple of {ncomp}"
        )
    return stream_len // ncomp - TAIL


def _detect_rate_k(stream_len: int) -> tuple[str, int]:
    """Auto-detect ``(rate, K)`` from a stream length for standard block lengths.

    All CCSDS block lengths × rates yield distinct stream lengths
    (``NCOMP[rate] * (K + TAIL)``), so the pair is uniquely determined.

    Args:
        stream_len: Encoded stream length in bits.

    Returns:
        The unique ``(rate, K)`` matching *stream_len*.

    Raises:
        ValueError: If no standard block length matches.
    """
    for K in STANDARD_K:
        for rate, ncomp in NCOMP.items():
            if ncomp * (K + TAIL) == stream_len:
                return rate, K
    raise ValueError(
        f"Unrecognized Turbo stream length {stream_len}: "
        "not a CCSDS block length (1784/3568/7136/8920/16384); pass rate="
    )


def decode(punctured_bits: list[int], iterations: int = 5, rate: str | None = None) -> list[int]:
    """Decode a Turbo stream back into the payload bits.

    The stream format is detected from its length unless ``rate`` is given:
    the length ``NCOMP[rate] * (K + TAIL)`` uniquely identifies ``(rate, K)``
    for all CCSDS block lengths, so ``rate`` only needs to be passed for
    non-standard block lengths.  Input bits are hard 0/1 symbols (LLR scale
    ±1.0, positive = bit 0); erasures (any other value) map to LLR 0.0.

    Args:
        punctured_bits: Encoded bit stream (each element 0 or 1).
        iterations: Number of turbo iterations (default 5); ``0`` skips the
            MAP loop and returns the systematic bits unchanged.
        rate: Optional explicit rate string (``"1/2"``, ``"1/3"``, ``"1/4"``
            or ``"1/6"``); bypasses length-based detection.

    Returns:
        The recovered payload bits.
    """
    stream = list(punctured_bits)
    if not stream:
        return []
    if rate is None:
        rate, K = _detect_rate_k(len(stream))
    else:
        if rate not in NCOMP:
            raise ValueError(f"Unsupported Turbo code rate: {rate}")
        K = _k_for_rate(len(stream), rate)
    if iterations <= 0:
        return stream[0::NCOMP[rate]][:K]
    rx = _llr_array(stream)
    return _turbo_decode_core(rx, rate, K, iterations).tolist()


def decode_unpunctured(turbo_bits: list[int], iterations: int = 3) -> list[int]:
    """Decode a full rate‑1/3 (unpunctured) Turbo stream.

    A rate‑1/3 frame has length ``3 * (K + 4)``; since ``K`` is a multiple of
    8, the frame length is ``3K + 12 ≡ 12 (mod 24)`` and has 4 trailing
    padding bits when packed into whole bytes.  The true length is the unique
    candidate ``T`` in ``[len-7, len]`` satisfying ``T > 24`` and
    ``T ≡ 12 (mod 24)``.

    Args:
        turbo_bits: Encoded bit stream (possibly byte-padded).
        iterations: Number of turbo iterations (default 3).

    Returns:
        The recovered payload bits.
    """
    n = len(turbo_bits)
    for k in range(8):
        t = n - k
        if t > 24 and (t - 12) % 24 == 0:
            return decode(turbo_bits[:t], iterations=iterations, rate="1/3")
    raise ValueError("Unrecognized rate-1/3 stream: no valid length within the last 8 bits")


def decode_padded_rate16(bits: list[int]) -> list[int]:
    """Decode a byte-padded rate‑1/6 stream (CLI helper).

    A rate‑1/6 frame has length ``6 * (K + 4)``; since ``K`` is a multiple of
    8, the frame length is ``6K + 24 ≡ 24 (mod 48)`` and always a multiple of
    8 bits.  Callers that packed the stream into whole bytes may append up to
    7 trailing padding bits, so the true length is the unique candidate
    ``T`` in ``[len-7, len]`` satisfying ``T > 24`` and ``T ≡ 24 (mod 48)``.
    """
    n = len(bits)
    for k in range(8):
        t = n - k
        if t > 24 and (t - 24) % 48 == 0:
            return decode(bits[:t], rate="1/6")
    raise ValueError("Unrecognized rate-1/6 stream: no valid length within the last 8 bits")
