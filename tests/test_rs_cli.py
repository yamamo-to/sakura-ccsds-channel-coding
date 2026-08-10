import os, subprocess, sys
import pytest

VENV_PY = sys.executable  # use current python interpreter

@pytest.mark.parametrize("depth", [1, 2, 5])
def test_cli_rs_depth_roundtrip(depth):
    payload = os.urandom(1000)
    enc = subprocess.run(
        [VENV_PY, "-m", "ccsds_codec", "rs-enc", "--depth", str(depth)],
        input=payload,
        capture_output=True,
    )
    assert enc.returncode == 0, enc.stderr
    dec = subprocess.run(
        [VENV_PY, "-m", "ccsds_codec", "rs-dec", "--depth", str(depth)],
        input=enc.stdout,
        capture_output=True,
    )
    assert dec.returncode == 0, dec.stderr
    assert dec.stdout[: len(payload)] == payload

def test_cli_rs_depth_invalid_choice():
    proc = subprocess.run(
        [VENV_PY, "-m", "ccsds_codec", "rs-enc", "--depth", "6"],
        input=b"",
        capture_output=True,
    )
    assert proc.returncode != 0
    assert b"invalid choice" in proc.stderr
