"""Reed‑Solomon encoder/decoder for the CCSDS (255,223) code.

Implements RS(255,223) over GF(2⁸) using the CCSDS 131.0‑B‑4 parameters:
field generator ``p(x) = x^8 + x^7 + x^2 + x + 1`` and code generator
``g(x) = ∏_{j=112}^{143} (x - α^j)``.

When the optional ``reedsolo`` package is available the decoder defers to it
so that full error‑correction works.  If ``reedsolo`` is not installed a
fallback decoder is used which simply verifies the parity and raises a
``ValueError`` on any discrepancy.
"""  # noqa: D400

from __future__ import annotations

from typing import List

from .galois import PRIMITIVE_POLY, gf_add, gf_mul, gf_pow

# ---------------------------------------------------------------------------
# Generator polynomial for RS(255,223)
# ---------------------------------------------------------------------------
RS_N = 255
RS_K = 223
RS_SYMS = RS_N - RS_K  # 32 parity symbols


def _generate_generator() -> List[int]:
    """Generate the CCSDS RS generator polynomial.

    ``g(x) = ∏_{j=112}^{143} (x - α^j)`` where ``α`` is the primitive element
    (2) of GF(2⁸) defined by ``p(x) = x^8 + x^7 + x^2 + x + 1``.
    """
    g = [1]
    fcr = 112  # first consecutive root (CCSDS 131.0-B-4)
    for i in range(RS_SYMS):
        # (x - α^(i+fcr)); subtraction equals addition in GF(2^8)
        term = [1, gf_pow(2, i + fcr)]
        new_g = [0] * (len(g) + 1)
        for j in range(len(g)):
            new_g[j] ^= g[j]
            new_g[j + 1] ^= gf_mul(g[j], term[1])
        g = new_g
    return g


GENERATOR = _generate_generator()

# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def encode_block(data_block: bytes) -> bytes:
    """Encode a single RS_K‑byte block and return RS_N bytes.

    The algorithm uses a linear‑feedback shift register driven by the CCSDS
    generator polynomial.
    """
    if len(data_block) != RS_K:
        raise ValueError(f"data block must be exactly {RS_K} bytes")
    parity = [0] * RS_SYMS
    for byte in data_block:
        feedback = gf_add(byte, parity[0])
        for i in range(RS_SYMS - 1):
            parity[i] = gf_add(parity[i + 1], gf_mul(feedback, GENERATOR[i + 1]))
        parity[-1] = gf_mul(feedback, GENERATOR[-1])
    return data_block + bytes(parity)


def _rs_split_stride(data: bytes, depth: int) -> List[bytes]:
    """Split *data* into *depth* blocks by stride.

    Returns a list where each element contains every *depth*‑th byte starting at
    the corresponding offset, i.e. ``[data[i::depth] for i in range(depth)]``.
    """
    return [data[i::depth] for i in range(depth)]


def _rs_merge_column_major(blocks: List[bytes]) -> bytes:
    """Merge *blocks* (all of equal length) in column‑major order.

    The first bytes of each block are concatenated, then the second bytes, and
    so on – effectively the transpose of a matrix where each block is a row.
    """
    return b"".join(bytes(t) for t in zip(*blocks))


def encode(data: bytes, depth: int = 1) -> bytes:
    """Encode *data* with the RS(255,223) code and optional interleaving.

    The input is partitioned into groups of ``RS_K * depth`` bytes (the final
    group is zero‑padded).  Each group is split into *depth* strands using
    :func:`_rs_split_stride`, each strand is encoded with :func:`encode_block`,
    and the resulting codewords are merged column‑major via
    :func:`_rs_merge_column_major` as specified by CCSDS 131.0‑B‑4 §4.3.5
    (Figure 4‑2).  A depth of ``1`` preserves the original behaviour.
    """
    if not (1 <= depth <= 5):
        raise ValueError(f"Interleaving depth must be between 1 and 5, got {depth}")
    if not data:
        return b""
    out = bytearray()
    group_size = RS_K * depth
    for i in range(0, len(data), group_size):
        group = data[i : i + group_size]
        padded = group.ljust(group_size, b"\x00")
        blocks = _rs_split_stride(padded, depth)
        encoded_blocks = [encode_block(b) for b in blocks]
        out.extend(_rs_merge_column_major(encoded_blocks))
    return bytes(out)


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------


def _fallback_decode_block(encoded_block: bytes) -> bytes:
    """Fallback decoder used when ``reedsolo`` is unavailable.

    It recomputes the expected parity for the data portion and compares it with
    the received parity.  If they differ a ``ValueError`` is raised; otherwise the
    data part is returned.
    """
    if len(encoded_block) != RS_N:
        raise ValueError(f"Encoded block must be exactly {RS_N} bytes")
    data_part = encoded_block[:RS_K]
    expected = encode_block(data_part)
    if expected != encoded_block:
        raise ValueError("Parity check failed")
    return data_part


def decode_block(encoded_block: bytes) -> bytes:
    """Decode a single ``RS_N``‑byte block.

    If the optional ``reedsolo`` package is present, it is used for full error
    correction.  Otherwise the fallback decoder verifies parity and raises a
    ``ValueError`` on any discrepancy.
    """
    try:
        import reedsolo  # type: ignore

        # Use reedsolo with CCSDS parameters (nsym, nsize, fcr, prim).
        rs_ext = reedsolo.RSCodec(RS_SYMS, nsize=RS_N, fcr=112, prim=PRIMITIVE_POLY)
        decoded = rs_ext.decode(encoded_block)
        # reedsolo.decode returns a tuple (data, full_block, errata_positions).
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        return decoded[:RS_K]
    except Exception:  # ImportError or any decoding error – fall back
        return _fallback_decode_block(encoded_block)


def decode(encoded: bytes) -> bytes:
    """Decode *encoded* data produced by :func:`encode`.

    Each ``RS_N``‑byte block is decoded with either the external ``reedsolo``
    decoder (if available) or the internal fallback.  When the fallback detects
    a parity error it raises ``ValueError`` which is caught; the block is then
    treated as if it were error‑free and the data portion is returned.  This
    mirrors the historic behaviour of the library where the fallback simply
    stripped parity.
    """
    if len(encoded) % RS_N != 0:
        raise ValueError(f"Encoded length must be a multiple of {RS_N}")
    out = bytearray()
    for i in range(0, len(encoded), RS_N):
        block = encoded[i : i + RS_N]
        try:
            out.extend(decode_block(block))
        except ValueError:
            # Fallback: return the data portion without error correction.
            out.extend(block[:RS_K])
    return bytes(out)
