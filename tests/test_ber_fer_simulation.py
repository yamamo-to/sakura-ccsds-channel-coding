import numpy as np
import pytest

from ccsds_codec.conv import encode as conv_encode, viterbi_decode as conv_decode
from ccsds_codec.turbo import encode as turbo_encode
from ccsds_codec.core.turbo import _turbo_decode_core


# Helper: AWGN channel for BPSK mapping (0 -> +1, 1 -> -1)
def awgn_llr(
    bits: np.ndarray, eb_n0_db: float, rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Map bits to LLRs after AWGN.

    Args:
        bits: uint8 array of 0/1 bits.
        eb_n0_db: Eb/N0 in dB.
        rate: Code rate R (bits per channel symbol).
        rng: NumPy random generator.
    Returns:
        LLR array (float64) where positive -> bit 0, negative -> bit 1.
    """
    # BPSK symbols: 0 -> +1, 1 -> -1
    symbols = 1.0 - 2.0 * bits.astype(np.float64)
    sigma = np.sqrt(1.0 / (2.0 * rate * (10.0 ** (eb_n0_db / 10.0))))
    noise = rng.normal(0.0, sigma, size=symbols.shape)
    y = symbols + noise
    llr = 2.0 * y / (sigma ** 2)
    return llr


def simulate_conv(eb_n0_db: float, frames: int, rng_seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(rng_seed)
    total_bit_errors = 0
    total_frame_errors = 0
    # Warm‑up JIT compilation
    _ = conv_decode([1, -1], rate="1/2")
    for _ in range(frames):
        bits = rng.integers(0, 2, size=64, dtype=np.uint8)
        enc = conv_encode(bits.tolist(), rate="1/2")
        llr = awgn_llr(np.array(enc, dtype=np.uint8), eb_n0_db, rate=0.5, rng=rng)
        dec = conv_decode(llr.tolist(), rate="1/2")
        dec_arr = np.array(dec, dtype=np.uint8)
        bit_err = np.count_nonzero(dec_arr != bits)
        total_bit_errors += bit_err
        total_frame_errors += int(bit_err != 0)
    ber = total_bit_errors / (frames * 64)
    fer = total_frame_errors / frames
    return ber, fer


def simulate_turbo(eb_n0_db: float, frames: int, rng_seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(rng_seed)
    total_bit_errors = 0
    total_frame_errors = 0
    # Warm‑up JIT for turbo core (use correct length array)
    K_warm = 8
    N_warm = K_warm + 4  # TAIL = 4
    dummy_len = 3 * N_warm  # NCOMP["1/3"] = 3
    _ = _turbo_decode_core(np.ones(dummy_len, dtype=np.float64), rate="1/3", K=K_warm, iterations=1)
    K = 64  # payload length (multiple of 8)
    for _ in range(frames):
        bits = rng.integers(0, 2, size=K, dtype=np.uint8)
        enc = turbo_encode(bits.tolist(), rate="1/3")
        llr = awgn_llr(np.array(enc, dtype=np.uint8), eb_n0_db, rate=1 / 3, rng=rng)
        dec = _turbo_decode_core(llr, rate="1/3", K=K, iterations=3)
        dec_arr = np.asarray(dec, dtype=np.uint8)
        bit_err = np.count_nonzero(dec_arr != bits)
        total_bit_errors += bit_err
        total_frame_errors += int(bit_err != 0)
    ber = total_bit_errors / (frames * K)
    fer = total_frame_errors / frames
    return ber, fer


@pytest.mark.parametrize("frames", [500])
def test_conv_ber_fer_monotonic(frames):
    ber_low, fer_low = simulate_conv(2.0, frames, rng_seed=123)
    ber_high, fer_high = simulate_conv(5.0, frames, rng_seed=123)
    assert ber_high < ber_low
    assert fer_high < fer_low
    assert ber_high < 1e-2


@pytest.mark.parametrize("frames", [200])
def test_turbo_ber_fer_monotonic(frames):
    ber_low, fer_low = simulate_turbo(0.0, frames, rng_seed=456)
    ber_high, fer_high = simulate_turbo(2.5, frames, rng_seed=456)
    assert ber_high < ber_low
    assert fer_high < fer_low
    assert ber_high < 2e-2
