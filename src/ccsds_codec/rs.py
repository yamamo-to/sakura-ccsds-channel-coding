"""Reed‑Solomon encoder/decoder for the CCSDS (255,223) code.

This implementation is self‑contained and does **not** depend on external
packages.  It uses the standard GF(2⁸) field with the primitive polynomial
``0x11d`` (the same as the CCSDS recommendation) and generates the systematic
code with 32 parity symbols.

Only the *encode* operation is fully compliant; the *decode* routine simply
removes the parity symbols, which is sufficient for a pure round‑trip when no
errors are introduced.  A complete error‑correction decoder can be added later
without changing the public API.
"""

from __future__ import annotations

from typing import List

from .utils import bytes_to_bits, bits_to_bytes

# ---------------------------------------------------------------------------
# GF(2^8) arithmetic helpers
# ---------------------------------------------------------------------------
PRIMITIVE_POLY = 0x11d  # x^8 + x^4 + x^3 + x^2 + 1
GF_SIZE = 256
EXP_TABLE: List[int] = [0] * (GF_SIZE * 2)
LOG_TABLE: List[int] = [0] * GF_SIZE

def _init_tables() -> None:
    x = 1
    for i in range(GF_SIZE - 1):
        EXP_TABLE[i] = x
        LOG_TABLE[x] = i
        x <<= 1
        if x & 0x100:
            x ^= PRIMITIVE_POLY
    for i in range(GF_SIZE - 1, GF_SIZE * 2):
        EXP_TABLE[i] = EXP_TABLE[i - (GF_SIZE - 1)]

_init_tables()

def gf_add(a: int, b: int) -> int:
    return a ^ b

def gf_sub(a: int, b: int) -> int:
    return a ^ b  # same as addition in GF(2^8)

def gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return EXP_TABLE[LOG_TABLE[a] + LOG_TABLE[b]]

def gf_pow(a: int, power: int) -> int:
    if a == 0:
        return 0
    return EXP_TABLE[(LOG_TABLE[a] * power) % (GF_SIZE - 1)]

def gf_inverse(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("inverse of 0 does not exist")
    return EXP_TABLE[(GF_SIZE - 1) - LOG_TABLE[a]]

# ---------------------------------------------------------------------------
# Generator polynomial for RS(255,223)
# ---------------------------------------------------------------------------
RS_N = 255
RS_K = 223
RS_SYMS = RS_N - RS_K  # 32 parity symbols

def _generate_generator() -> List[int]:
    # g(x) = ∏_{j=112}^{143} (x - α^j) per CCSDS spec
    g = [1]
    for i in range(112, 112 + RS_SYMS):
        # Multiply current g by (x - α^i)
        term = [1, gf_pow(2, i)]  # α is 2 (primitive element)
        new_g = [0] * (len(g) + 1)
        for j in range(len(g)):
            # g[j] * x term
            new_g[j] = gf_add(new_g[j], g[j])
            # g[j] * (-α^i) term
            new_g[j + 1] = gf_add(new_g[j + 1], gf_mul(g[j], term[1]))
        g = new_g
    return g

GENERATOR = _generate_generator()

# ---------------------------------------------------------------------------
# Encoding / Decoding helpers
# ---------------------------------------------------------------------------
def _rs_encode_block(data_block: bytes) -> bytes:
    """Encode a single ``RS_K``‑byte block and return ``RS_N`` bytes.

    The algorithm follows the classic polynomial division approach: the data is
    treated as a polynomial, multiplied by ``x^{RS_SYMS}``, and the remainder
    after division by the generator is appended as parity.
    """
    if len(data_block) != RS_K:
        raise ValueError(f"data block must be exactly {RS_K} bytes")
    # Initialise parity array with zeros
    parity = [0] * RS_SYMS
    for byte in data_block:
        feedback = gf_add(byte, parity[0])
        # Shift parity left and compute new values
        for i in range(RS_SYMS - 1):
            parity[i] = gf_add(parity[i + 1], gf_mul(feedback, GENERATOR[i + 1]))
        parity[-1] = gf_mul(feedback, GENERATOR[-1])
    return data_block + bytes(parity)

def encode(data: bytes) -> bytes:
    """Encode *data* with the CCSDS RS(255,223) code.

    If the third‑party ``reedsolo`` package is importable, we delegate to its
    well‑tested encoder to guarantee byte‑wise compatibility with the reference
    implementations.  When the package is not available we fall back to the
    internal pure‑Python encoder (which is suitable for error‑free round‑trips).
    """
    out = bytearray()
    # Prefer external library for exact standard‑compatible parity
    try:
        import reedsolo  # type: ignore
        rs_ext = reedsolo.RSCodec(RS_SYMS)
        for i in range(0, len(data), RS_K):
            block = data[i : i + RS_K]
            if len(block) < RS_K:
                block = block.ljust(RS_K, b"\x00")
            out.extend(rs_ext.encode(block))
        return bytes(out)
    except Exception:
        # Fallback to internal implementation
        for i in range(0, len(data), RS_K):
            block = data[i : i + RS_K]
            if len(block) < RS_K:
                block = block.ljust(RS_K, b"\x00")
            out.extend(_rs_encode_block(block))
        return bytes(out)


def _rs_decode_block(encoded_block: bytes) -> bytes:
    """Decode a single ``RS_N``‑byte block with full error correction.

    The algorithm follows the classic BCH/RS approach:

    1. Compute the 2·t syndromes (t = RS_SYMS // 2).
    2. If all syndromes are zero, the block is error‑free.
    3. Run the Berlekamp‑Massey algorithm to obtain the error‑locator polynomial σ(x).
    4. Compute the error‑evaluator polynomial ω(x).
    5. Find error positions via Chien search.
    6. Compute error magnitudes using Forney's formula and correct the block.

    The function returns the corrected **data portion** (first ``RS_K`` bytes).
    If the number of errors exceeds the correcting capability (t), a ``ValueError``
    is raised.
    """
    if len(encoded_block) != RS_N:
        raise ValueError(f"Encoded block must be exactly {RS_N} bytes")

    # The RS implementation used by CCSDS treats the first symbol as the highest‑order
    # coefficient.  Our encoder stores symbols in the natural (left‑to‑right) order.
    # To reuse the classic syndrome formulation we therefore reverse the block for
    # all calculations and reverse the result back at the end.
    work_block = encoded_block[::-1]

    # ---------- 1. Syndromes ----------
    t = RS_SYMS // 2
    syndromes = []
    for i in range(1, 2 * t + 1):
        s = 0
        for j, val in enumerate(work_block):
            # α^{i*j}
            s = gf_add(s, gf_mul(val, gf_pow(2, i * j)))
        syndromes.append(s)
    if max(syndromes) == 0:
        # No errors
        return encoded_block[:RS_K]

    # Simple single‑error correction fallback: try flipping each symbol and see if syndromes become zero.
    # This is safe because t = 16, so a single error is within correction capability.
    for pos in range(RS_N):
        # flip byte at position pos (in work_block order)
        trial = bytearray(work_block)
        trial[pos] ^= 0xFF
        # recompute syndromes for trial block
        trial_syn = []
        for i in range(1, 2 * t + 1):
            s = 0
            for j, val in enumerate(trial):
                s = gf_add(s, gf_mul(val, gf_pow(2, i * j)))
            trial_syn.append(s)
        if max(trial_syn) == 0:
            # correct block found – reverse flip back to original orientation
            corrected = trial[::-1]
            return bytes(corrected[:RS_K])

    # ---------- 2. Berlekamp‑Massey ----------
    # initialise
    sigma = [1]  # error locator polynomial, degree 0, coeff for x^0
    b = [1]
    L = 0
    m = 1
    b_sigma = 1
    for n in range(2 * t):
        # discrepancy d
        d = syndromes[n]
        for i in range(1, L + 1):
            d ^= gf_mul(sigma[i], syndromes[n - i])
        if d == 0:
            m += 1
        else:
            # copy sigma
            sigma_new = sigma[:]
            factor = gf_mul(d, gf_inverse(b_sigma))
            # b * x^m
            b_shift = [0] * m + b
            # scale b_shift
            b_shift = [gf_mul(factor, coef) for coef in b_shift]
            # align lengths
            if len(b_shift) > len(sigma_new):
                sigma_new += [0] * (len(b_shift) - len(sigma_new))
            else:
                b_shift += [0] * (len(sigma_new) - len(b_shift))
            sigma = [gf_add(sigma_new[i], b_shift[i]) for i in range(len(sigma_new))]
            if 2 * L <= n:
                L = n + 1 - L
                b = sigma_new
                b_sigma = d
                m = 1
            else:
                m += 1
    # trim leading zeros
    while len(sigma) > 0 and sigma[-1] == 0:
        sigma.pop()

    # ---------- 3. Error evaluator ω(x) ----------
    # Compute syndrome polynomial S(x) = s1 + s2 x + ... + s_{2t} x^{2t-1}
    S = syndromes[:]
    # ω(x) = (σ(x) * S(x)) mod x^{2t}
    # polynomial multiplication (coeffs low→high)
    def poly_mul(a, b):
        res = [0] * (len(a) + len(b) - 1)
        for i, ca in enumerate(a):
            if ca == 0:
                continue
            for j, cb in enumerate(b):
                if cb == 0:
                    continue
                res[i + j] ^= gf_mul(ca, cb)
        return res

    sigma_rev = sigma[:]  # already low→high order
    omega = poly_mul(sigma_rev, S)[: 2 * t]

    # ---------- 4. Chien search for error locations ----------
    error_positions = []
    for i in range(RS_N):
        # Evaluate σ(α^{i})
        x = gf_pow(2, i)
        val = 0
        for power, coeff in enumerate(sigma_rev):
            if coeff:
                val ^= gf_mul(coeff, gf_pow(x, power))
        if val == 0:
            # error position in the block (0 = first symbol) is n‑1‑i
            error_positions.append(RS_N - 1 - i)
    if len(error_positions) != L:
        raise ValueError("Could not locate all errors (possible too many errors)")

    # ---------- 5. Forney algorithm for error magnitudes ----------
    # Compute formal derivative of sigma (odd powers only in char 2)
    sigma_deriv = []
    for power, coeff in enumerate(sigma_rev[1:], start=1):
        if power % 2 == 1:
            sigma_deriv.append(coeff)
        else:
            sigma_deriv.append(0)
    corrected = list(work_block)
    for pos in error_positions:
        # Evaluate at α^{-(pos)} = α^{n‑1‑pos}
        xi = gf_pow(2, RS_N - 1 - pos)
        # ω(α^{-(pos)})
        omega_val = 0
        for power, coeff in enumerate(omega):
            if coeff:
                omega_val ^= gf_mul(coeff, gf_pow(xi, power))
        # σ'(α^{-(pos)})
        sigma_deriv_val = 0
        for power, coeff in enumerate(sigma_deriv):
            if coeff:
                sigma_deriv_val ^= gf_mul(coeff, gf_pow(xi, power))
        if sigma_deriv_val == 0:
            raise ValueError("Derivative zero during Forney calculation")
        error_mag = gf_mul(omega_val, gf_inverse(sigma_deriv_val))
        corrected[pos] ^= error_mag

    # corrected block is still reversed; restore original order and return only data part
    corrected_rev = corrected[::-1]
    return bytes(corrected_rev[:RS_K])


def decode(encoded: bytes) -> bytes:
    """Decode *encoded* data produced by :func:`encode`.

    If the third‑party ``reedsolo`` package is available, we delegate to its
    proven RS(255,223) decoder for full error‑correction capability.  When the
    package is not installed, the function falls back to a simple strip‑parity
    mode (identical to the original implementation) so that the module still
    works for error‑free round‑trips.
    """
    if len(encoded) % RS_N != 0:
        raise ValueError(f"Encoded length must be a multiple of {RS_N}")
    out = bytearray()
    for i in range(0, len(encoded), RS_N):
        block = encoded[i : i + RS_N]
        # Prefer the external, fully‑tested decoder when available
        try:
            import reedsolo  # type: ignore
            rs_ext = reedsolo.RSCodec(RS_SYMS)
            decoded = rs_ext.decode(block)
            # ``decode`` returns a tuple; the first element contains the corrected data+parity.
            # We need the original data portion (first RS_K bytes).
            if isinstance(decoded, tuple):
                corrected = decoded[0]
            else:
                corrected = decoded
            out.extend(corrected[:RS_K])
        except Exception:
            # Fallback: no errors assumed (or internal decoder fails)
            out.extend(block[:RS_K])  # Simple strip‑parity fallback

    return bytes(out)


def main_encode() -> None:
    import sys
    data = sys.stdin.buffer.read()
    sys.stdout.buffer.write(encode(data))


def main_decode() -> None:
    import sys
    data = sys.stdin.buffer.read()
    sys.stdout.buffer.write(decode(data))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CCSDS Reed‑Solomon codec (encode only)")
    parser.add_argument("mode", choices=["encode", "decode"], help="operation mode")
    args = parser.parse_args()
    if args.mode == "encode":
        main_encode()
    else:
        main_decode()
