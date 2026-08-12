"""Performance benchmarks for the CCSDS codec.

Covers all three codec families (Convolutional, Reed-Solomon, Turbo) with both
JIT-compiled (numba) and offline modes.  The goal is to ensure that compute-
intensive kernels stay within reasonable bounds on a typical CI runner and that
numba offline mode (``NUMBA_DISABLE_JIT``) still produces correct output.
"""

import random
import subprocess
import sys

from ccsds_codec.core.convolutional import decode as conv_decode, encode as conv_encode
from ccsds_codec.core.reed_solomon import (
    RS_K,
    decode as rs_decode,
    encode_block as rs_encode,
)


def _random_bits(length: int) -> list[int]:
    return [random.randint(0, 1) for _ in range(length)]


def test_conv_encode_decode_performance():
    """Convolutional encode + Viterbi decode 1 000 bits in under 5 ms."""
    bits = _random_bits(1_000)
    encoded = conv_encode(bits)
    conv_decode(encoded)  # warm JIT
    import time

    t0 = time.perf_counter()
    decoded = conv_decode(encoded)
    t1 = time.perf_counter()
    assert decoded == bits
    duration = t1 - t0
    assert duration < 0.005, f"Conv decode took {duration * 1000:.1f} ms"


def test_rs_encode_decode_performance():
    """RS(255, 223) encode + decode a full block in under 20 ms."""
    data = bytes(range(RS_K))
    encoded = rs_encode(data)
    rs_decode(encoded)  # warm JIT
    import time

    t0 = time.perf_counter()
    decoded = rs_decode(encoded)
    t1 = time.perf_counter()
    assert decoded == data
    duration = t1 - t0
    assert duration < 0.02, f"RS encode+decode took {duration * 1000:.1f} ms"


def _run_numba_offline_perf_tests() -> subprocess.CompletedProcess:
    """Execute performance tests in a subprocess with numba disabled.

    Returns the ``subprocess.CompletedProcess`` so the caller can inspect
    stdout/stderr on failure.
    """
    script = """\
import random
import time

from ccsds_codec.core.convolutional import decode as conv_decode, encode as conv_encode
from ccsds_codec.core.reed_solomon import RS_K, decode as rs_decode, encode_block as rs_encode

# --- Convolutional decode (numba offline) ---
bits = [random.randint(0, 1) for _ in range(1000)]
encoded = conv_encode(bits)
t0 = time.perf_counter()
decoded = conv_decode(encoded)
t1 = time.perf_counter()
assert decoded == bits, "conv offline correctness failed"
conv_ms = (t1 - t0) * 1000

# --- Turbo decode (numba offline) ---
from ccsds_codec.core.turbo import decode as turbo_decode, encode as turbo_encode

bits2 = [random.randint(0, 1) for _ in range(1000)]
enc2 = turbo_encode(bits2, puncture=True)
t2 = time.perf_counter()
dec2 = turbo_decode(enc2, rate="1/2")
t3 = time.perf_counter()
assert dec2 == bits2, "turbo offline correctness failed"
turbo_ms = (t3 - t2) * 1000

# --- RS encode+decode (numba offline — pure Python) ---
data = bytes(range(RS_K))
enc3 = rs_encode(data)
t4 = time.perf_counter()
dec3 = rs_decode(enc3)
t5 = time.perf_counter()
assert dec3 == data, "rs offline correctness failed"
rs_ms = (t5 - t4) * 1000

print(f"conv_decode={conv_ms:.1f}ms turbo_decode={turbo_ms:.1f}ms rs_encode_decode={rs_ms:.1f}ms")
"""
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**dict(__import__("os").environ), "NUMBA_DISABLE_JIT": "1"},
    )


def test_conv_decode_numba_offline():
    """Verify conv decode works correctly and completes in < 500 ms when numba is disabled."""
    result = _run_numba_offline_perf_tests()
    assert result.returncode == 0, f"numba offline perf test failed:\n{result.stderr}"
    line = result.stdout.strip().split("\n")[-1]
    # Extract conv_decode time from "conv_decode=XXXms turbo_decode=... rs=..."
    ms = float(line.split("conv_decode=")[1].split("ms")[0])
    assert ms < 500, f"conv decode (numba offline) took {ms:.1f} ms, exceeds 500 ms"


def test_turbo_decode_numba_offline():
    """Verify turbo decode works correctly and completes in < 1 000 ms when numba is disabled."""
    result = _run_numba_offline_perf_tests()
    assert result.returncode == 0, f"numba offline perf test failed:\n{result.stderr}"
    line = result.stdout.strip().split("\n")[-1]
    # Extract turbo_decode time
    ms = float(line.split("turbo_decode=")[1].split("ms")[0])
    assert ms < 1000, f"turbo decode (numba offline) took {ms:.1f} ms, exceeds 1000 ms"
