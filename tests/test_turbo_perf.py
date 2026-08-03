"""Performance benchmark for the CCSDS Turbo codec.

The test generates a random payload of 1 000 bits, encodes it with puncturing
(rate 1/4), and measures the time required for a full Log‑MAP decode with three
iterations. The goal is to keep the decode time below 0.5 seconds on a typical
CI runner.
"""

import random
import time

from ccsds_codec.turbo import decode, encode


def _random_bits(length: int) -> list[int]:
    return [random.randint(0, 1) for _ in range(length)]


def test_decode_performance():
    bits = _random_bits(1_000)
    punctured = encode(bits, puncture=True)
    start = time.perf_counter()
    decoded = decode(punctured, iterations=3)
    duration = time.perf_counter() - start
    # Basic sanity check – decoded payload must match original
    assert decoded == bits
    # Performance threshold (seconds)
    assert duration < 0.5, f"Turbo decode took {duration:.3f}s, exceeds 0.5s"
