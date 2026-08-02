"""簡易ベンチマークスクリプト – エンコード/デコード速度測定\n\nこのスクリプトは `src/ccsds_codec` の主要関数について、\n- Reed‑Solomon エンコード/デコード\n- Convolutional エンコード/ Viterbi デコード\n- Turbo エンコード/パンクチャ、簡易デコード\nのスループット (Mbps) を測ります。\n\n実行例: ``python -m scripts.benchmark``\n"""

import time
import numpy as np
from ccsds_codec import utils, conv, rs, turbo

def _measure(func, *args, repeats: int = 5):
    # warm‑up
    func(*args)
    start = time.perf_counter()
    for _ in range(repeats):
        func(*args)
    elapsed = time.perf_counter() - start
    return elapsed / repeats

def bench_rs(block_len: int = 1024):
    data = np.random.bytes(block_len)
    t_enc = _measure(rs.encode, data)
    encoded = rs.encode(data)
    t_dec = _measure(rs.decode, encoded)
    mbps_enc = (block_len * 8) / (t_enc * 1e6)
    mbps_dec = (block_len * 8) / (t_dec * 1e6)
    print(f"RS ({block_len} B): encode {mbps_enc:.2f} Mbps, decode {mbps_dec:.2f} Mbps")

def bench_conv(bits_len: int = 1024):
    bits = utils.bytes_to_bits(np.random.bytes(bits_len))
    t_enc = _measure(conv.encode, bits)
    encoded = conv.encode(bits)
    t_vit = _measure(conv.viterbi_decode, encoded)
    mbps_enc = (len(bits) / t_enc) / 1e6 * 8
    mbps_vit = (len(bits) / t_vit) / 1e6 * 8
    print(f"CONV ({bits_len} B): encode {mbps_enc:.2f} Mbps, Viterbi {mbps_vit:.2f} Mbps")

def bench_turbo(bits_len: int = 1024):
    bits = utils.bytes_to_bits(np.random.bytes(bits_len))
    t_enc = _measure(turbo.encode, bits, puncture=False)
    t_enc_p = _measure(turbo.encode, bits, puncture=True)
    encoded_full = turbo.encode(bits, puncture=False)
    encoded_p = turbo.encode(bits, puncture=True)
    t_dec = _measure(turbo.decode_unpunctured, encoded_full)
    t_dec_p = _measure(turbo.decode, encoded_p, iterations=3)
    mbps_enc = (len(bits) / t_enc) / 1e6 * 8
    mbps_enc_p = (len(bits) / t_enc_p) / 1e6 * 8
    mbps_dec = (len(bits) / t_dec) / 1e6 * 8
    mbps_dec_p = (len(bits) / t_dec_p) / 1e6 * 8
    print(f"Turbo ({bits_len} B): encode {mbps_enc:.2f} Mbps (full), {mbps_enc_p:.2f} Mbps (punctured)\n        decode {mbps_dec:.2f} Mbps (full), {mbps_dec_p:.2f} Mbps (punctured)")

if __name__ == "__main__":
    bench_rs()
    bench_conv()
    bench_turbo()
