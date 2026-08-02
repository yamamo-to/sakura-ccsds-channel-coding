# CCSDS Turbo Code – Specification Summary (Rate 1/3)

## 1. 標準概要
| 項目 | 内容 | 標準での記述（章/節） |
|------|------|----------------------|
| **符号ブロック長** | 4 の倍数であることが必須。許容ブロック長は **1784, 3568, 7136, 8920, 16384 ビット**。 | 3.1.1 *Block Lengths* |
| **ターミネーション（Tail bits）** | 各 RSC コンstituent コーダは **K‑1 = 4** ビットの 0 で埋めた tail を付与し、状態を 0 にリセット。デコーダはこの tail により終了状態を判定できる。 | 3.2.3 *Termination* |
| **生成多項式** | - Feedback (母) : `g₀ = 10011₂ (23₈)`<br>- Forward (前方) : `g₁ = 11011₂ (33₈)`<br>両方とも **K = 5**（constraint length）。 | 3.3.1 *Constituent Encoder Polynomials* |
| **コードレート** | 基本は **1/3**（Systematic + Parity₁ + Parity₂）。パンクチャにより **1/2, 2/3, 3/4, 5/6, 7/8** が得られる。パンクチャは *Rate 1/4*（Systematic + Parity₁ + punctured Parity₂）方式で、Parity₂ の奇数位置ビットは除去される。 | 3.4.1 *Code Rates* |
| **パンクチャパターン（Rate 1/4）** | - Systematic : すべて送信<br>- Parity₁   : すべて送信<br>- Parity₂   : **偶数インデックス**（0, 2, 4,…）のみ送信<br>出力長は `2·L + ⌈L/2⌉` ビット。 | 3.4.2 *Puncturing Pattern* |
| **インタリーバ** | Quadratic‑Permutation（表 5‑1 に定義）。
```
π(i) = (f₁·i + f₂·i²) mod K   (i = 0…K‑1)
```
`f₁` と `f₂` はブロックサイズごとに決まる。例（K が 8 の倍数の場合）: `f₁ = 17, f₂ = 31`。インタリーバは **自己逆** ではなく、デ‑インタリーバが別途必要。 | 5.2.1 *Quadratic‑Permutation Interleaver* |
| **デコーダ** | Log‑MAP（BCJR）を 2 本の RSC デコーダで交互に実行。
1. a‑priori = 0 で開始。
2. 第 1 コンシステュエント（G0）で BCJR → extrinsic LLR を取得。
3. 取得した extrinsic をインタリーバで入れ替えて第 2 コンシステュエント（G1）へ。
4. 第 2 コンシステュエントからの extrinsic をデ‑インタリーバし、次イテレーションの a‑priori とする。
5. 指定イテレーション数（例: 5）後、Systematic LLR と最終 a‑priori を足し合わせて硬決定（LLR≥0 → 0, それ以外 → 1）。 | 6.1 – 6.4 *Iterative MAP Decoding* |
| **LLR スケーリング** | LLR = +α がビット `0` の尤度、LLR = –α がビット `1` の尤度。実装上は **0.5 ~ 1.0** 程度のスケールが数値安定性で推奨される。 | 6.2.2 *LLR Normalisation* |
| **エラーパターン上限** | RSC の自由距離は 5。Turbo の自由距離はインタリーバ次第で 4 ~ 6 程度。訂正可能ビット数は `t = (d_free‑1)/2`。低 SNR ではソフト入力 MAP が必須。 | 4.2 *Error‑Correction Capability* |
| **テストベクトル** | 付録 A に **4 つのブロック長**（1784, 3568, 7136, 8920）に対する Systematic / Parity₁ / Parity₂ の 8 bit 単位ベクトルが掲載され、実装検証に利用できる。 | 付録 A – “Reference Test Vectors” |

---

## 2. 実装上の必須ポイント

### 2.1 入力長の 4 の倍数化（Tail bits）
```python
def _pad_to_multiple_of_4(bits: List[int]) -> List[int]:
    """CCSDS で要求される 4 の倍数に tail (0) を付加する。"""
    remainder = len(bits) % 4
    if remainder:
        bits = bits + [0] * (4 - remainder)   # tail bits are zeros
    return bits
```
- **エンコード** 前に呼び出し、`len(bits) % 4 == 0` を保証する。
- エンコード後は tail ビットを **情報ビット長** として保持し、デコード側で除去できるように長さ情報を伝える（例: `(encoded, tail_len)`）

### 2.2 インタリーバ／デ‑インタリーバ
```python
INTERLEAVER_TABLE = {
    # K : (f1, f2)   – CCSDS 付録表 5‑1 の抜粋例
    8:   (17, 31),
    16:  (17, 31),
    32:  (17, 31),
    # … 必要に応じて全ブロック長を記述 …
}

def interleaver(bits: List[int]) -> List[int]:
    K = len(bits)
    f1, f2 = INTERLEAVER_TABLE[K]
    out = [0] * K
    for i in range(K):
        j = (f1 * i + f2 * i * i) % K
        out[j] = bits[i]
    return out

def deinterleaver(bits: List[int]) -> List[int]:
    K = len(bits)
    f1, f2 = INTERLEAVER_TABLE[K]
    out = [0] * K
    for i in range(K):
        j = (f1 * i + f2 * i * i) % K
        out[i] = bits[j]
    return out
```
- **自己逆** ではなく必ず逆写像（デ‑インタリーバ）を実装することが重要。

### 2.3 パンクチャ / デパンクチャ
```python
def puncture(sys: List[int], p1: List[int], p2: List[int]) -> List[int]:
    # Parity2 の偶数位置だけ残す（Rate 1/4）
    p2_filt = [p2[i] for i in range(len(p2)) if i % 2 == 0]
    return sys + p1 + p2_filt

def depuncture(punct: List[int]) -> Tuple[List[int], List[int], List[int]]:
    L = payload_len_from_punctured(len(punct))
    sys = punct[:L]
    p1  = punct[L:2*L]
    filt = punct[2*L:]
    p2 = []
    f_idx = 0
    for i in range(L):
        if i % 2 == 0:
            p2.append(filt[f_idx]); f_idx += 1
        else:
            p2.append(0)              # missing bits → LLR 0 (erasure)
    return sys, p1, p2
```
- `payload_len_from_punctured` は CCSDS で規定された式 `2·L + ⌈L/2⌉ = p_len` を解くユーティリティ。

### 2.4 BCJR（Log‑MAP）実装の注意点
```python
neg_inf = float('-inf')

def _bcjr(sys_llr: List[float], parity_llr: List[float], generator: int) -> List[float]:
    N = len(sys_llr)
    # α, β テーブル初期化
    alpha = [{0: 0.0}] + [{s: neg_inf for s in range(1 << K)} for _ in range(N)]
    beta  = [{s: neg_inf for s in range(1 << K)} for _ in range(N + 1)]
    beta[N][0] = 0.0

    # 前向き（α）再帰
    for i in range(N):
        for state in range(1 << K):
            prev = alpha[i].get(state, neg_inf)
            if prev == neg_inf:
                continue
            for u in (0, 1):
                ns = ((state << 1) | u) & MASK
                parity_bit = _parity(ns & generator)
                bm = (sys_llr[i] * (1 - 2 * u) + parity_llr[i] * (1 - 2 * parity_bit)) / 2.0
                metric = prev + bm
                alpha[i + 1][ns] = _logsumexp(alpha[i + 1][ns], metric)

    # 後向き（β）再帰
    for i in range(N - 1, -1, -1):
        for state in range(1 << K):
            best = neg_inf
            for u in (0, 1):
                ns = ((state << 1) | u) & MASK
                parity_bit = _parity(ns & generator)
                bm = (sys_llr[i] * (1 - 2 * u) + parity_llr[i] * (1 - 2 * parity_bit)) / 2.0
                metric = beta[i + 1][ns] + bm
                best = _logsumexp(best, metric)
            beta[i][state] = best

    # ポステリア LLR 計算
    post: List[float] = []
    for i in range(N):
        L0 = neg_inf
        L1 = neg_inf
        for state in range(1 << K):
            for u in (0, 1):
                ns = ((state << 1) | u) & MASK
                parity_bit = _parity(ns & generator)
                bm = (sys_llr[i] * (1 - 2 * u) + parity_llr[i] * (1 - 2 * parity_bit)) / 2.0
                prob = alpha[i].get(state, neg_inf) + bm + beta[i + 1][ns]
                if u == 0:
                    L0 = _logsumexp(L0, prob)
                else:
                    L1 = _logsumexp(L1, prob)
        post.append(L0 - L1)
    return post
```
- **`neg_inf`** (= `float('-inf')`) を未定義状態の表現に使用し、`_logsumexp` が数値的に安定するようにする。
- **LLR のスケーリング**は 0.5‑1.0 程度に抑えると、`exp` のオーバーフロー/アンダーフローが防げる。

### 2.5 イテレーティブ MAP 手順（擬似コード）
```python
apriori = [0.0] * N
for _ in range(iterations):
    # 1st constituent (G0)
    sys1 = [s + a for s, a in zip(sys_llr, apriori)]
    post1 = _bcjr(sys1, p1_llr, G0)
    extrinsic1 = [post1[i] - sys1[i] for i in range(N)]
    inter_extr = interleaver(extrinsic1)

    # 2nd constituent (G1) – systematic LLR は同じだがインタリーバ適用
    sys2 = [s + a for s, a in zip(interleaver(sys_llr), inter_extr)]
    post2 = _bcjr(sys2, p2_llr, G1)
    extrinsic2 = [post2[i] - sys2[i] for i in range(N)]
    apriori = deinterleaver(extrinsic2)   # 次イテレーションへの a‑priori

# 最終硬決定
final_llr = [sys_llr[i] + apriori[i] for i in range(N)]
hard = [0 if llr >= 0 else 1 for llr in final_llr]
```
- `interleaver` / `deinterleaver` は **自己逆ではない** ことに注意。
- `iterations` は 5 程度が標準的だが、要求性能に応じて増減可能。

---

## 3. テストベクトルの活用方法

付録 A に掲載されている **4 つのブロック長** のベクトルは次の流れで検証できる。
```python
from ccsds_codec.turbo import encode, decode, interleaver, deinterleaver
from ccsds_codec.utils import bytes_to_bits, bits_to_bytes

# 例: 1784‑bit ブロック（付録ベクトルを bytes で取得）
plain_bytes = bytes.fromhex('…')   # 付録 A の Systematic 部
plain_bits = bytes_to_bits(plain_bytes)

# エンコード（Rate 1/3）
enc = encode(plain_bits, puncture=False)
# パンクチャ（Rate 1/4）
pun = encode(plain_bits, puncture=True)

# デコード（パンクチャ版）
rec = decode(pun, iterations=5)
assert rec == plain_bits
```
- パンクチャ後の長さは `2*L + ceil(L/2)` になることを `payload_len_from_punctured` でチェック。
- インタリーバとデ‑インタリーバの**恒等性**は `deinterleaver(interleaver(x)) == x` で必ず確認。

---

## 4. まとめ
- **ブロック長は 4 の倍数**（tail bits で調整）。
- **インタリーバ** は quadratic‑permutation、**デ‑インタリーバ** が必要。
- **パンクチャ** は Parity₂ の奇数位置ビットを除去し、復元時は 0（LLR=0）で埋める。
- **BCJR（Log‑MAP）** は α/β の `-inf` 初期化と `_logsumexp` の利用で数値安定性を保つ。
- 標準付録の **テストベクトル** を用いたラウンドトリップテストが実装正当性の最も簡易かつ信頼できる手段。

この仕様まとめをもとに、CCSDS‑準拠の **Turbo エンコーダ/デコーダ** を実装・検証してください。
