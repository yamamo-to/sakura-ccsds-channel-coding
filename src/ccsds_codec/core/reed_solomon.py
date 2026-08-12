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

from .galois import PRIMITIVE_POLY, gf_add, gf_mul, gf_pow, gf_from_dual_basis, gf_to_dual_basis

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
# Cached reedsolo.RSCodec instances (same parameters, constructed once)
# ---------------------------------------------------------------------------
_rs_codec: object | None = None
_rs_codec_dual: object | None = None


def _get_rs_codec(dual_basis: bool) -> object | None:
    """Return a cached reedsolo.RSCodec instance, or None if unavailable."""
    global _rs_codec, _rs_codec_dual
    if dual_basis:
        if _rs_codec_dual is None:
            try:
                import reedsolo  # type: ignore
                _rs_codec_dual = reedsolo.RSCodec(RS_SYMS, nsize=RS_N, fcr=112, prim=PRIMITIVE_POLY)
            except ImportError:
                return None
        return _rs_codec_dual
    if _rs_codec is None:
        try:
            import reedsolo  # type: ignore
            _rs_codec = reedsolo.RSCodec(RS_SYMS, nsize=RS_N, fcr=112, prim=PRIMITIVE_POLY)
        except ImportError:
            return None
    return _rs_codec


def _clear_rs_codec_cache() -> None:
    """Clear cached reedsolo.RSCodec instances (for testing only)."""
    global _rs_codec, _rs_codec_dual
    _rs_codec = None
    _rs_codec_dual = None


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def encode_block(data_block: bytes, *, dual_basis: bool = False) -> bytes:
    """Encode a single RS_K‑byte block and return RS_N bytes.

    The algorithm uses a linear‑feedback shift register driven by the CCSDS
    generator polynomial.

    Args:
        data_block: ``RS_K`` bytes to encode.
        dual_basis: When ``True``, encode in conventional basis first, then
            transform the **entire** ``RS_N``‑byte codeword (data + parity) to
            dual‑basis representation.  This preserves full compatibility with
            ``reedsolo`` for error correction.

    Returns:
        Encoded ``RS_N`` bytes.
    """
    if len(data_block) != RS_K:
        raise ValueError(f"data block must be exactly {RS_K} bytes")
    parity = [0] * RS_SYMS
    for byte in data_block:
        feedback = gf_add(byte, parity[0])
        for i in range(RS_SYMS - 1):
            parity[i] = gf_add(parity[i + 1], gf_mul(feedback, GENERATOR[i + 1]))
        parity[-1] = gf_mul(feedback, GENERATOR[-1])
    result = data_block + bytes(parity)
    if dual_basis:
        result = bytes(gf_to_dual_basis(b) for b in result)
    return result


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


def encode(data: bytes, depth: int = 1, *, dual_basis: bool = False) -> bytes:
    """Encode *data* with the RS(255,223) code and optional interleaving.

    The input is partitioned into groups of ``RS_K * depth`` bytes (the final
    group is zero‑padded).  Each group is split into *depth* strands using
    :func:`_rs_split_stride`, each strand is encoded with :func:`encode_block`,
    and the resulting codewords are merged column‑major via
    :func:`_rs_merge_column_major` as specified by CCSDS 131.0‑B‑4 §4.3.5
    (Figure 4‑2).  A depth of ``1`` preserves the original behaviour.

    Args:
        data: Raw bytes to encode (arbitrary length; the final partial group
            is zero‑padded to ``RS_K * depth`` bytes).
        depth: Interleaving depth, 1..5 (default 1).
        dual_basis: When ``True``, encode each data symbol in dual‑basis
            representation (CCSDS 131.0‑B‑4 §4.1 note).

    Returns:
        Encoded bytes: ``RS_N * depth`` bytes per input group of
        ``RS_K * depth`` bytes.
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
        encoded_blocks = [encode_block(b, dual_basis=dual_basis) for b in blocks]
        out.extend(_rs_merge_column_major(encoded_blocks))
    return bytes(out)


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------


def _fallback_decode_block(encoded_block: bytes, *, dual_basis: bool = False) -> bytes:
    """Fallback decoder used when ``reedsolo`` is unavailable.

    It recomputes the expected parity for the data portion and compares it with
    the received parity.  If they differ a ``ValueError`` is raised; otherwise the
    data part is returned.

    Args:
        encoded_block: ``RS_N`` encoded bytes.
        dual_basis: When ``True``, the data portion of *encoded_block* is
            already in dual‑basis representation.  Parity is verified by
            calling :func:`encode_block` **without** dual‑basis transformation
            (the data must already be in the form the LFSR expects).
    """
    if len(encoded_block) != RS_N:
        raise ValueError(f"Encoded block must be exactly {RS_N} bytes")
    data_part = encoded_block[:RS_K]
    expected = encode_block(data_part)
    if expected != encoded_block:
        raise ValueError("Parity check failed")
    return data_part


def decode_block(encoded_block: bytes, *, dual_basis: bool = False) -> bytes:
    """Decode a single ``RS_N``‑byte block.

    If the optional ``reedsolo`` package is present, it is used for full error
    correction.  Otherwise the fallback decoder verifies parity and raises a
    ``ValueError`` on any discrepancy.

    When ``dual_basis`` is ``True``:
    1. Transform every byte of the received codeword from dual‑basis to
       conventional‑basis.
    2. Use ``reedsolo`` for full error correction on the conventional codeword.
    3. Convert the decoded data portion back to dual‑basis.

    Args:
        encoded_block: ``RS_N`` encoded bytes.
        dual_basis: When ``True``, the input is in dual‑basis representation.
            Full error correction is supported by converting to conventional
            basis, decoding with ``reedsolo``, then converting back.

    Returns:
        Decoded ``RS_K`` bytes.  With ``dual_basis=True`` the result is in
        dual‑basis representation.
    """
    if dual_basis:
        try:
            conv = bytes(gf_from_dual_basis(b) for b in encoded_block)
            rs_ext = _get_rs_codec(dual_basis=True)
            if rs_ext is None:
                return _fallback_decode_block(encoded_block, dual_basis=dual_basis)
            decoded = rs_ext.decode(conv)
            if isinstance(decoded, tuple):
                decoded = decoded[0]
            return bytes(gf_to_dual_basis(b) for b in decoded[:RS_K])
        except Exception:  # ReedSolomonError or decode error
            return _fallback_decode_block(encoded_block, dual_basis=dual_basis)
    rs_ext = _get_rs_codec(dual_basis=False)
    if rs_ext is None:
        return _fallback_decode_block(encoded_block)
    try:
        decoded = rs_ext.decode(encoded_block)
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        return bytes(decoded[:RS_K])
    except Exception:  # ReedSolomonError or decode error
        return _fallback_decode_block(encoded_block)


def decode(encoded: bytes, depth: int = 1, *, dual_basis: bool = False) -> bytes:
    """Decode *encoded* data produced by :func:`encode` with interleaving.

    The *depth* argument must be an integer between 1 and 5 (inclusive).  The
    input is interpreted as a sequence of interleaved RS codewords, each group
    consisting of ``RS_N * depth`` bytes (the result of ``encode`` with the same
    *depth*).  For each group we split the column‑major interleaving back into
    *depth* individual codewords using :func:`_rs_split_stride`, decode each
    codeword with :func:`decode_block` (or fall back to the data portion on
    ``ValueError``), and finally merge the decoded data blocks column‑major via
    :func:`_rs_merge_column_major`.  This implements CCSDS 131.0‑B‑4 §4.3.5
    (Figure 4‑2).  A *depth* of ``1`` preserves the original behaviour.

    Args:
        encoded: Encoded bytes; length must be a multiple of ``RS_N * depth``.
        depth: Interleaving depth, 1..5 (default 1).
        dual_basis: When ``True``, the input is in dual‑basis representation.
            Each decoded data block is converted from dual‑basis back to
            conventional‑basis.

    Returns:
        Decoded bytes: ``RS_K * depth`` per input group, including any zero
        padding from the encoder.  Compare ``dec[:len(original)]`` against the
        original data.
    """
    if not (1 <= depth <= 5):
        raise ValueError(f"Interleaving depth must be between 1 and 5, got {depth}")
    if not encoded:
        return b""
    if len(encoded) % (RS_N * depth) != 0:
        raise ValueError(f"Encoded length must be a multiple of {RS_N * depth}")
    out = bytearray()
    group_size = RS_N * depth
    for i in range(0, len(encoded), group_size):
        group = encoded[i : i + group_size]
        # split interleaved bytes into individual encoded blocks
        blocks = _rs_split_stride(group, depth)
        decoded_blocks = []
        for block_idx, block in enumerate(blocks):
            try:
                decoded_blocks.append(decode_block(block, dual_basis=dual_basis))
            except ValueError as e:
                # Provide context about which block failed.
                group_idx = i // group_size
                # Raise with context about block and group indices.
                raise ValueError(
                    f"Failed to decode block {block_idx} in group {group_idx}: {e}"
                ) from e
        # merge decoded data blocks to reconstruct original (possibly padded) data
        out.extend(_rs_merge_column_major(decoded_blocks))
    return bytes(out)
