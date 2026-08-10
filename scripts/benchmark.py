"""簡易ベンチマークスクリプト – エンコード/デコード速度測定

このスクリプトは `src/ccsds_codec` の主要関数について、
- Reed‑Solomon エンコード/デコード
- Convolutional エンコード/ Viterbi デコード
- Turbo エンコード/パンクチャ、簡易デコード
のスループット (Mbps) を測ります。

実行例: ``python -m scripts.benchmark``
"""

import time
from functools import partial

import numpy as np
from ccsds_codec import conv, rs, turbo, utils


def _measure(func, repeats: int = 5):
    """Measure average wall-clock time of a zero-argument callable."""
    func()
    start = time.perf_counter()
    for _ in range(repeats):
        func()
    elapsed = time.perf_counter() - start
    return elapsed / repeats


def bench_rs(block_len: int = 1024):
    data = np.random.bytes(block_len)
    t_enc = _measure(partial(rs.encode, data))
    encoded = rs.encode(data)
    t_dec = _measure(partial(rs.decode, encoded))
    mbps_enc = (block_len * 8) / (t_enc * 1e6)
    mbps_dec = (block_len * 8) / (t_dec * 1e6)
    print(f"RS ({block_len} B): encode {mbps_enc:.2f} Mbps, decode {mbps_dec:.2f} Mbps")


def bench_conv(bits_len: int = 1024):
    bits = utils.bytes_to_bits(np.random.bytes(bits_len))
    t_enc = _measure(partial(conv.encode, bits))
    encoded = conv.encode(bits)
    t_vit = _measure(partial(conv.viterbi_decode, encoded))
    mbps_enc = (len(bits) / t_enc) / 1e6 * 8
    mbps_vit = (len(bits) / t_vit) / 1e6 * 8
    print(f"CONV ({bits_len} B): encode {mbps_enc:.2f} Mbps, Viterbi {mbps_vit:.2f} Mbps")


def bench_turbo(bits_len: int = 1024):
    bits = utils.bytes_to_bits(np.random.bytes(bits_len))
    t_enc = _measure(partial(turbo.encode, bits, rate="1/3"))
    t_enc_p = _measure(partial(turbo.encode, bits, rate="1/2"))
    encoded_full = turbo.encode(bits, rate="1/3")
    encoded_p = turbo.encode(bits, rate="1/2")
    t_dec = _measure(partial(turbo.decode_unpunctured, encoded_full))
    t_dec_p = _measure(partial(turbo.decode, encoded_p, iterations=3, rate="1/2"))
    mbps_enc = (len(bits) / t_enc) / 1e6 * 8
    mbps_enc_p = (len(bits) / t_enc_p) / 1e6 * 8
    mbps_dec = (len(bits) / t_dec) / 1e6 * 8
    mbps_dec_p = (len(bits) / t_dec_p) / 1e6 * 8
    print(
        f"Turbo ({bits_len} B): encode {mbps_enc:.2f} Mbps (full), "
        f"{mbps_enc_p:.2f} Mbps (punctured)\n"
        f"        decode {mbps_dec:.2f} Mbps (full), {mbps_dec_p:.2f} Mbps (punctured)"
    )


if __name__ == "__main__":
    bench_rs()
    bench_conv()
    bench_turbo()
