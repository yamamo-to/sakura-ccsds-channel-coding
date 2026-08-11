import os, subprocess, sys
from pathlib import Path
import pytest

VENV_PY = sys.executable  # use current python interpreter

# The CLI subprocess runs `python -m ccsds_codec` in the current interpreter,
# which may not have the package installed (pytest.ini adds src/ only for the
# pytest process itself).  Prepending the repo's src/ directory to PYTHONPATH
# keeps the CLI importable from any interpreter.
_SRC_PATH = Path(__file__).resolve().parents[1] / "src"


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_SRC_PATH) + (os.pathsep + existing if existing else "")
    return env


@pytest.mark.parametrize("depth", [1, 2, 5])
def test_cli_rs_depth_roundtrip(depth):
    payload = os.urandom(1000)
    enc = subprocess.run(
        [VENV_PY, "-m", "ccsds_codec", "rs-enc", "--depth", str(depth)],
        input=payload,
        capture_output=True,
        env=_cli_env(),
    )
    assert enc.returncode == 0, enc.stderr
    dec = subprocess.run(
        [VENV_PY, "-m", "ccsds_codec", "rs-dec", "--depth", str(depth)],
        input=enc.stdout,
        capture_output=True,
        env=_cli_env(),
    )
    assert dec.returncode == 0, dec.stderr
    assert dec.stdout[: len(payload)] == payload

def test_cli_rs_depth_invalid_choice():
    proc = subprocess.run(
        [VENV_PY, "-m", "ccsds_codec", "rs-enc", "--depth", "6"],
        input=b"",
        capture_output=True,
        env=_cli_env(),
    )
    assert proc.returncode != 0
    assert b"invalid choice" in proc.stderr
