"""In-process tests for the unified command-line interface (ccsds_codec.cli)."""

import io
import sys

import pytest

from ccsds_codec import cli
from ccsds_codec.core.bits import bits_to_bytes, bytes_to_bits
from ccsds_codec.core.convolutional import encode as conv_encode
from ccsds_codec.core.reed_solomon import encode as rs_encode


class _FakeStdin:
    """Minimal stand-in for sys.stdin exposing a ``buffer`` attribute."""

    def __init__(self, data: bytes) -> None:
        self.buffer = io.BytesIO(data)


class _FakeStdout:
    """Minimal stand-in for sys.stdout exposing a ``buffer`` attribute."""

    def __init__(self) -> None:
        self.buffer = io.BytesIO()


def _run_cli(monkeypatch: pytest.MonkeyPatch, args: list[str], stdin: bytes = b"") -> bytes:
    """Run ``cli.main()`` with *args* and return everything written to stdout."""
    monkeypatch.setattr(sys, "argv", ["ccsds_codec", *args])
    monkeypatch.setattr(sys, "stdin", _FakeStdin(stdin))
    stdout = _FakeStdout()
    monkeypatch.setattr(sys, "stdout", stdout)
    cli.main()
    return stdout.buffer.getvalue()


def test_rs_enc_dec_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    """rs-enc followed by rs-dec recovers the payload (bytes in/out)."""
    # A full interleaving group: len % (223 * depth) == 0 avoids zero padding.
    payload = bytes(i % 256 for i in range(223 * 5))
    encoded = _run_cli(monkeypatch, ["rs-enc", "--depth", "5"], payload)
    assert encoded == rs_encode(payload, depth=5)
    assert _run_cli(monkeypatch, ["rs-dec", "--depth", "5"], encoded) == payload


def test_rs_dec_reports_corruption(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A malformed RS stream aborts with exit code 1 and a diagnostic."""
    with pytest.raises(SystemExit) as excinfo:
        _run_cli(monkeypatch, ["rs-dec"], b"\x00" * 3)
    assert excinfo.value.code == 1
    assert "RS decode error" in capsys.readouterr().err


def test_conv_enc_dec_roundtrip_rate12(monkeypatch: pytest.MonkeyPatch) -> None:
    """conv-enc/conv-dec round-trip at the default rate 1/2."""
    payload = bytes([0, 1, 1, 0, 1, 0, 0, 1] * 3)
    encoded = _run_cli(monkeypatch, ["conv-enc", "--rate", "1/2"], payload)
    assert encoded == bits_to_bytes(conv_encode(bytes_to_bits(payload), rate="1/2"))
    assert _run_cli(monkeypatch, ["conv-dec", "--rate", "1/2"], encoded) == payload


def test_conv_enc_dec_roundtrip_rate23(monkeypatch: pytest.MonkeyPatch) -> None:
    """Punctured rate 2/3 round-trips through the byte-padded decoder."""
    payload = bytes(range(16))
    encoded = _run_cli(monkeypatch, ["conv-enc", "--rate", "2/3"], payload)
    assert _run_cli(monkeypatch, ["conv-dec", "--rate", "2/3"], encoded) == payload


def test_turbo_enc_dec_roundtrip_rate16(monkeypatch: pytest.MonkeyPatch) -> None:
    """turbo-enc/turbo-dec round-trip at rate 1/6 (byte-aligned frames)."""
    payload = bytes([0, 1, 1, 0, 1, 0, 0, 1] * 4)
    encoded = _run_cli(monkeypatch, ["turbo-enc", "--rate", "1/6"], payload)
    assert _run_cli(monkeypatch, ["turbo-dec", "--rate", "1/6"], encoded) == payload


def test_turbo_enc_dec_roundtrip_rate13(monkeypatch: pytest.MonkeyPatch) -> None:
    """turbo-enc/turbo-dec round-trip at rate 1/3 (unpunctured)."""
    payload = bytes([1, 0, 0, 1, 1, 0, 1, 0] * 4)
    encoded = _run_cli(monkeypatch, ["turbo-enc", "--rate", "1/3"], payload)
    assert _run_cli(monkeypatch, ["turbo-dec", "--rate", "1/3"], encoded) == payload


def test_turbo_enc_dec_roundtrip_rate12(monkeypatch: pytest.MonkeyPatch) -> None:
    """turbo-enc/turbo-dec round-trip at rate 1/2 (punctured, explicit rate)."""
    payload = bytes([0, 1, 1, 0, 1, 0, 0, 1] * 4)
    encoded = _run_cli(monkeypatch, ["turbo-enc", "--rate", "1/2"], payload)
    assert _run_cli(monkeypatch, ["turbo-dec", "--rate", "1/2"], encoded) == payload


def test_turbo_dec_autodetects_rate13(monkeypatch: pytest.MonkeyPatch) -> None:
    """turbo-dec without --rate detects the rate-1/3 stream length (unpunctured path)."""
    payload = bytes([1, 0, 0, 1, 1, 0, 1, 0] * 4)
    encoded = _run_cli(monkeypatch, ["turbo-enc", "--rate", "1/3"], payload)
    assert _run_cli(monkeypatch, ["turbo-dec"], encoded) == payload


def test_turbo_dec_autodetects_rate12(monkeypatch: pytest.MonkeyPatch) -> None:
    """turbo-dec without --rate detects a standard-length rate-1/2 stream."""
    payload = bytes(range(223))  # 1784 bits: a CCSDS standard block length
    encoded = _run_cli(monkeypatch, ["turbo-enc", "--rate", "1/2"], payload)
    assert _run_cli(monkeypatch, ["turbo-dec"], encoded) == payload


def test_rand_is_self_inverse(monkeypatch: pytest.MonkeyPatch) -> None:
    """The randomizer mode restores the input when applied twice."""
    data = bytes(range(64))
    scrambled = _run_cli(monkeypatch, ["rand"], data)
    assert scrambled != data
    assert _run_cli(monkeypatch, ["rand"], scrambled) == data


def test_invalid_mode_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown mode aborts with a usage error."""
    with pytest.raises(SystemExit):
        _run_cli(monkeypatch, ["bogus"], b"")


def test_invalid_rate_is_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """An unsupported rate aborts with a usage error."""
    with pytest.raises(SystemExit):
        _run_cli(monkeypatch, ["conv-enc", "--rate", "9/9"], b"\x00")
    assert "invalid rate" in capsys.readouterr().err
