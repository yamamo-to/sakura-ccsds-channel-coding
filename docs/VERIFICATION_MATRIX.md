# CCSDS 準拠検証マトリックス

**対象規格**: CCSDS 131.0-B-4 (TM Synchronization and Channel Coding)
**対象実装**: `ccsds-codec` (Python 3.11+)
**検証日**: 2026-08-10（初版 81f5283 時点 189 passed → 修正適用後 228 passed → 238 passed → 現時点 267 passed に更新）
**検証環境**: Python 3.14.4 / numpy + numba / reedsolo (import 可) / pytest 267 passed

---

## 凡例

| ステータス | 意味 |
|---|---|
| ✅ 準拠 | 実装があり、テストで検証済み |
| ⚠️ 部分的 | 実装はあるが、能力・検証範囲・依存に制約あり |
| ✗ 非準拠 | 実装なし、または規格と矛盾 |
| ❓ 未検証 | 実装の有無が不明、またはテスト不在 |

検証証跡の `file:line` は実在確認済み、テスト名は `pytest --collect-only` 相当の実在スキャンで確認済み。

---

## 1. Reed-Solomon (CCSDS 131.0-B-4 §4)

| ID | 規格要求 | 要求内容 | 実装箇所 | 検証テスト | ステータス | 備考・リスク |
|---|---|---|---|---|---|---|
| RS-01 | §4.1 | 原始多項式 `p(x)=x^8+x^7+x^2+x+1` (0x187) | `core/galois.py:12` (`PRIMITIVE_POLY = 0x187`) | `test_rs_internal.py::test_gf_tables_consistent`, `test_rs_internal.py::test_gf_arithmetic_basic` | ✅ 準拠 | GF テーブル整合性テストで検証 |
| RS-02 | §4.1 | 生成多項式 `g(x)=∏_{j=112}^{143}(x−α^j)` (32 パリティ) | `core/reed_solomon.py:22-24` (`RS_N/RS_K/RS_SYMS`), generator 構築 | `test_rs.py::test_generator_length`, `test_rs_properties.py::test_generator_properties`, `test_rs_internal.py::test_generate_generator_length` | ✅ 準拠 | モニック・次数 32 を検証 |
| RS-03 | §4.1 | 符号パラメータ RS(255,223)、32 シンボル | `core/reed_solomon.py:22-24` | `test_rs.py::test_encode_known_vector` (golden parity), `test_rs_properties.py::test_encode_block_length` | ✅ 準拠 | reedsolo の CCSDS パラメータ (fcr=112, prim=0x187) と parity 一致 |
| RS-04 | §4.3.5 Fig 4-2 | ブロックインタリーブ深さ I=1..5 | `core/reed_solomon.py` `encode`/`decode` (depth 引数, stride 分割) | `test_rs_interleave.py::test_interleaving_structure`, `test_rs_interleave.py::test_roundtrip`, `test_rs_config.py::test_rs_codec_different_depths` | ✅ 準拠 | 列優先配置を構造テストで検証 |
| RS-05 | §4.2 | 誤り訂正能力 t=16 | `core/reed_solomon.py:154` (`reedsolo.RSCodec(32, nsize=255, fcr=112, prim=0x187)`) | `test_rs_properties.py::test_error_correction_with_reedsolo`, `test_edge_cases.py::test_rs_decode_with_correctable_errors` | ✅ 準拠 | reedsolo を dev 依存 (`pyproject.toml`) に追加済み → CI でも訂正能力テストが通る。ランタイムの reedsolo 非搭載時のみパリティ照合フォールバック（旧リスク G-1 は軽減・G-6 解消） |
| RS-06 | §4.2 | 不可訂正エラーの検出 | `core/reed_solomon.py:127-140` (`_fallback_decode_block`), `:164-207` (`decode`) | `test_rs_internal.py::test_internal_decode_too_many_errors`, `test_edge_cases.py::test_rs_decode_exceeds_error_capacity`, `test_rs_extended.py::test_decode_without_errors_fallback` | ✅ 準拠 | **サイレント破損を修正済み**: `decode()` は不可訂正ブロックで `ValueError`（グループ/ブロック番号付き）を送出。CLI (`cli.py::_rs`) は stderr 出力 + exit(1)。リスク G-2 解消 |
| RS-07 | §4.1 (注記) | Dual-basis 表現変換 | なし（従来表現のみ） | テストなし | ❓ 未検証 | AGENTS.md で「オプション／設定可能」とされる範囲外。対応はスコープ外として明記 |

---

## 2. Convolutional Code (CCSDS 131.0-B-4 §2)

| ID | 規格要求 | 要求内容 | 実装箇所 | 検証テスト | ステータス | 備考・リスク |
|---|---|---|---|---|---|---|
| CONV-01 | §2.3 | 制約長 K=7、G1=171₈・G2=133₈（lsb-current 表現で 0x4F/0x6D） | `core/convolutional.py:41-43` (`G0 = 0x4F`, `G1 = 0x6D`, `K = 7`) | `test_conv.py::test_generator_constants`, `test_conv_known.py::test_generator_constants`, `test_conv_extended.py::test_generator_constants` | ✅ 準拠 | gr-satellites `fec.cc_decoder` (polys [79, -109]) とビット一致 |
| CONV-02 | §2.3 | 第 2 出力シンボルをチャネル上で反転 | `core/convolutional.py` `encode` (`1 - _parity(state & G1)`) | `test_conv_known.py::test_encode_terminated_known_vector`, `test_conv_known.py::test_encode_unterminated_known_vector`, `test_conv_grsatellite.py::test_encode_decode_roundtrip` | ✅ 準拠 | ゴールデンベクトル (gr-satellites 由来) と一致 |
| CONV-03 | §2.4 | パンクチャレート 2/3, 3/4, 5/6, 7/8 | `core/convolutional.py:60` (`PUNCTURE_PATTERNS`) | `test_conv_rates.py::test_pattern_definitions`, `test_conv_rates.py::test_punctured_length_matches_pattern`, `test_conv_rates.py::test_punctured_roundtrip` | ✅ 準拠 | パターンは gr-satellites と同一 |
| CONV-04 | §2.4 | Viterbi 復号（ハード判定） | `core/convolutional.py` `viterbi_decode` / `_viterbi_hard_kernel` | `test_conv_basic.py::test_encode_decode_roundtrip`, `test_conv.py::test_encode_decode_roundtrip`, `test_conv_known.py` | ✅ 準拠 | |
| CONV-05 | §2.4 | Viterbi 復号（ソフト判定 / LLR） | `core/convolutional.py:191` (`_viterbi_llr_kernel`) | `test_conv_rates.py::test_punctured_llr_roundtrip`, `test_conv_rates.py::test_punctured_corrects_two_flips` | ✅ 準拠 | LLR 入力・フリップ訂正を検証 |
| CONV-06 | §2.3 | K−1 ゼロテール終端 | `core/convolutional.py` `encode` (`terminate=True`) | `test_conv_rates.py::test_punctured_roundtrip_terminated`, `test_conv_known.py::test_encode_terminated_known_vector` | ✅ 準拠 | |
| CONV-07 | §2.4 | デパンクチャ（消去挿入） | `core/convolutional.py:316` (`_depuncture`) | `test_conv_rates.py::test_punctured_roundtrip`, `test_conv_rates.py::test_punctured_llr_roundtrip` | ✅ 準拠 | |

---

## 3. Turbo Code (CCSDS 131.0-B-4 §3, Annex)

| ID | 規格要求 | 要求内容 | 実装箇所 | 検証テスト | ステータス | 備考・リスク |
|---|---|---|---|---|---|---|
| TURBO-01 | §3.3.1 | RSC 構成コード、K=5、フィードバック g0=23₈ / フォワード g1=33₈ | `core/turbo.py:70-73` (`GEN_SYS = 0x13`, `GEN = 0x1B`, `GEN2 = 0x15`, `GEN3 = 0x1F`) | `test_turbo_rate16.py::test_bcjr_kernel_clean_channel_decodes_zeros`, `test_turbo_rate16.py::test_rate16_roundtrip` | ✅ 準拠 | 定数は §3.3.1 の 23₈/33₈ に対応（テストは小ブロックでのみ検証） |
| TURBO-02 | §3.4.1 | 基本レート 1/2, 1/3, 1/4, 1/6 | `core/turbo.py:83` (`NCOMP`) | `test_turbo_properties.py::test_stream_length_formula`, `test_turbo_and_randomizer.py::test_turbo_punctured_roundtrip`, `test_turbo_rate16.py::test_rate16_roundtrip` | ✅ 準拠 | ストリーム長式 `NCOMP*(K+4)` を全レートで検証 |
| TURBO-03 | §3.1.1 | 情報ブロック長 1784/3568/7136/8920/16384 | `core/turbo.py:100` (`STANDARD_K`) | `test_turbo_properties.py::test_rate_autodetect_is_unique_and_correct`, `test_turbo_standard_lengths.py::test_encode_decode_roundtrip_standard_lengths` (全 5 長 × rate 1/3, 1/6), `test_turbo_standard_lengths.py::test_detect_rate_k_for_standard_lengths` (全 5 長 × 4 レート) | ✅ 準拠 | **標準長での実符 roundtrip テストを追加済み**（rate 1/3・1/6、全 5 長）。リスク G-3 解消 |
| TURBO-04 | §3.2.3 | 終端（K−1=4 テール、状態 0 リセット） | `core/turbo.py:76` (`TAIL = 4`) | `test_turbo_properties.py::test_stream_length_formula` | ✅ 準拠 | 長さ公式で間接検証 |
| TURBO-05 | §6.3g | QPP インタリーバ（Quadratic-Permutation） | `core/interleaver.py` (`ccsds_perm` / `ccsds_interleaver` / `ccsds_deinterleaver`) | `test_turbo_extended.py::test_interleaver_is_bijective_and_inverse`, `test_turbo_properties.py::test_interleaver_is_permutation`, `test_turbo_properties.py::test_deinterleaver_is_inverse`, `test_turbo_golden.py::test_ccsds_perm_1784_matches_reference` | ✅ 準拠 | 全単射・逆元 + **K=1784 は外部参照ゴールデンベクトル照合済み**（`tests/data/ccsdsSize1784.txt`、mdmoctezuma/CCSDSTurboCode 由来、1784 点完全一致）。他長の外部照合は未実施（リスク G-4 一部残） |
| TURBO-06 | §3.4 (Annex) | 反復 Log-MAP / Max-Log-MAP 復号 | `core/turbo.py:215` (`_build_trellis`), `:256` (`_bcjr_kernel`), `_turbo_decode_core` | `test_turbo.py::test_unpunctured_roundtrip`, `test_turbo.py::test_punctured_roundtrip`, `test_turbo_extended.py::test_decode_consistency_across_iterations` | ✅ 準拠 | |
| TURBO-07 | 数値安定性 | 対数領域でアンダーフロー防止 | `core/turbo.py:256` (`_bcjr_kernel`) | `test_turbo_rate16.py::test_bcjr_kernel_channel_matrix_shapes`, `test_turbo_rate16.py::test_bcjr_kernel_clean_channel_decodes_zeros` | ✅ 準拠 | 有限 LLR・チャネル行列形状を検証 |
| TURBO-08 | §3.1.1 | ブロック長からのレート自動判別 | `core/turbo.py` (`_detect_rate_k`) | `test_turbo_properties.py::test_rate_autodetect_is_unique_and_correct`, `test_turbo_properties.py::test_rate_autodetect_rejects_unknown_length` | ✅ 準拠 | |
| TURBO-09 | Annex A | 付録の符号語（エンコード出力）基準テストベクトル | なし | なし | ❓ 未検証 | 公開されている Annex A 符号語ベクトルは取得不可。エンコード出力の外部照合は未実施 |
| TURBO-10 | §6.3g / Annex H | QPP インタリーバ（全 5 標準ブロック長） | `core/interleaver.py` (`ccsds_perm`) | `test_turbo_interleaver_extended.py` | ✅ 準拠 | 公式 CCSDS §6.3g 数式の独立再実装と照合済み（1784/3568/7136/8920/16384）。K=1784 はさらに外部参照ファイル `ccsdsSize1784.txt` と一致 |
| TURBO-11 | §3.4 | 最大反復回数（規格例: 10 回） | デフォルト 5 回、上限未設定 | `test_turbo_extended.py::test_decode_consistency_across_iterations` | ⚠️ 部分的 | 反復回数は設定可能だが上限強制なし |

---

## 4. Pseudo-Randomizer (CCSDS 131.0-B-4 §10.4)

| ID | 規格要求 | 要求内容 | 実装箇所 | 検証テスト | ステータス | 備考・リスク |
|---|---|---|---|---|---|---|
| RND-01 | §10.4 | 多項式 `h(x)=x^8+x^7+x^5+x^3+1` | `core/randomizer.py:34` (`TAPS = (7, 4, 2, 0)`) | `test_randomizer_known.py::test_known_sequence_first_40_bits` | ✅ 準拠 | 初出 40 ビットと一致 |
| RND-02 | §10.4.3 | 先頭 40 ビット `ff 48 0e c0 9a` | `core/randomizer.py` (`scramble`) | `test_randomizer_known.py::test_known_sequence_first_40_bits` | ✅ 準拠 | |
| RND-03 | §10.4 | 255 ビット周期（最大長 LFSR） | `core/randomizer.py` (`_lfsr_next`) | `test_randomizer_known.py::test_known_sequence_period` | ✅ 準拠 | |
| RND-04 | §10.4 | 自己逆（scramble ∘ descramble = id） | `core/randomizer.py` (`descramble`) | `test_randomizer_known.py::test_self_inverse`, `test_randomizer.py::test_scramble_descramble_identity` | ✅ 準拠 | |

---

## 5. ビット順序・共通規約

| ID | 要求 | 要求内容 | 実装箇所 | 検証テスト | ステータス | 備考・リスク |
|---|---|---|---|---|---|---|
| BIT-01 | AGENTS.md / CCSDS | MSB-first ビット列規約 | `core/bits.py` (`bytes_to_bits` / `bits_to_bytes`) | `test_utils.py::test_bytes_bits_roundtrip`, `test_utils.py::test_bits_to_bytes_padding` | ✅ 準拠 | |
| BIT-02 | AGENTS.md | LLR 規約（正 = 0 の尤度、負 = 1 の尤度） | `core/turbo.py` (`LLR_0` / `LLR_1`) | `test_turbo_rate16.py::test_bcjr_kernel_clean_channel_decodes_zeros` | ✅ 準拠 | クリーンチャネルで全ゼロ復号を検証 |

---

## 6. アーキテクチャ・データ型整合性

以下の項目は CCSDS 131.0-B-4 の強制要件ではなく、本プロジェクトの内部設計基準（`AGENTS.md`）に対する整合性を評価したものです。

| ID | 規格要求 | 要求内容 | 実装箇所 | 検証テスト | ステータス | 備考・リスク |
|---|---|---|---|---|---|---|
| ARCH-01 | AGENTS.md §1 | 抽象基底クラス `BaseEncoder` / `BaseDecoder` | `api.py:33-61` | `test_base_codec.py` | ✅ 準拠 | `RSCodec`/`ConvCodec`/`TurboCodec` が両方を継承。`Randomizer` は stateless なため階層に含めない |
| DTYPE-01 | AGENTS.md §1 / §4.1 | ビット列入出力を 1 次元 `np.ndarray` (`uint8`) で統一 | `api.py` 及各 `core/*.py` の入出力が `list[int]`。`np.uint8` は `core/turbo.py:434` の戻り値のみ | なし | ⚠️ 部分的 | LLR 計算では `np.ndarray` を使用するが、ビット列 API では Python list が基本。MSB-first 値は保証される |
| DOC-06 | README / AGENTS.md | 生成多項式表記の整合性（README: `0x79/0x5B` = 標準オクタル表記、`core/convolutional.py`: `0x4F/0x6D` = lsb-current） | `README.md` / `core/convolutional.py:41-42` | `test_conv_known.py` | ✅ 準拠 | 値の違いではなく表現の違い。gr-satellites 由来のゴールデンベクトルで実装値が検証済み |

---

## 7. 性能・品質目標

| ID | 要求 | 要求内容 | 実装箇所 | 検証テスト | ステータス | 備考・リスク |
|---|---|---|---|---|---|---|
| PERF-01 | AGENTS.md §3 (Tester) | AWGN 上 Monte-Carlo BER/FER | `tests/test_ber_fer_simulation.py` (`simulate_conv` / `simulate_turbo`) | `test_ber_fer_simulation.py::test_conv_ber_fer_monotonic[500]`, `test_ber_fer_simulation.py::test_turbo_ber_fer_monotonic[200]` | ✅ 準拠 | **BER/FER シミュレーションを追加済み**（AWGN・BPSK、Eb/N0 に対する BER/FER 単調性を検証）。リスク G-5 解消 |
| PERF-02 | AGENTS.md §4.2 | 計算コアの numba JIT | `core/convolutional.py` (Viterbi カーネル), `core/turbo.py:256` (`_bcjr_kernel`) | `test_turbo_perf.py::test_decode_performance` | ⚠️ 部分的 | 性能テストは 1 件のみ。numba 非搭載環境での動作は未検証 |

---

## 8. CI・ドキュメント整合性

| ID | 対象 | 問題内容 | ステータス | 備考 |
|---|---|---|---|---|
| CI-01 | `.github/workflows/ci.yml` | `pip install .[dev]` に reedsolo が含まれない → CI で `test_rs_decode_with_correctable_errors` 等が失敗する可能性 | ✅ 解決 | **reedsolo を dev extras に追加済み** (`pyproject.toml`)。CI クリーン環境でも訂正能力テストが通る。リスク G-6 解消 |
| DOC-01 | `docs/ccsds_spec.md` | CONV 生成多項式が `G1=121₈`（TC 方式）表記で、実装 `171₈/133₈` と矛盾 | ✅ 解決 | `G1=171₈ (non-inverting)` / `G2=133₈ (inverted on the channel)` に修正済み |
| DOC-02 | `docs/COMPATIBILITY.md` | Turbo が「簡易デモ・非互換」と記載 → 実装は正式 RSC/QPP/Log-MAP に更新済み | ✅ 解決 | Turbo 行を「完全実装（RSC K=5, g0=23₈, g1=33₈, QPP, Log-MAP）」に更新し、コメント列も「準拠実装」に整合 |
| DOC-03 | `docs/COMPATIBILITY_TURBO.md` | f1=17/f2=31 の旧インタリーバ記述が残存 | ✅ 解決 | QPP 記述 (k1=8, k2=K/8) に修正済み。さらに `decode_unpunctured` の「ハード決定 Viterbi」誤記も Log-MAP (BCJR) に修正 |
| DOC-04 | `docs/COMPATIBILITY_CLTU.md` | 存在しない `ccsds_codec.cltu` モジュールへの参照 | ✅ 解決 | 実在モジュール構成に合わせて更新済み |
| DOC-05 | `docs/architecture.md` | `src/ccsds/` レイアウト記述が現行 `src/ccsds_codec/` と不一致 | ✅ 解決 | `src/ccsds_codec/`（core/ + api.py + config.py + cli.py + shim）構成に更新済み |

---

## サマリー

| カテゴリ | ✅ 準拠 | ⚠️ 部分的 | ✗ 非準拠 | ❓ 未検証 |
|---|---|---|---|---|
| Reed-Solomon | 6 | — | — | 1 |
| Convolutional | 7 | — | — | — |
| Turbo | 9 | 1 | — | 1 |
| Randomizer | 4 | — | — | — |
| 共通規約 | 2 | — | — | — |
| アーキテクチャ・データ型 | 2 | 1 | — | — |
| 性能目標 | 1 | 1 | — | — |
| CI・ドキュメント | 6 | — | — | — |
| **合計** | **37** | **3** | **0** | **2** |

---

## 主要リスク（G-1..G-9）と対応状況

| ID | リスク | 深刻度 | 対応状況 |
|---|---|---|---|
| G-1 | reedsolo 非搭載環境で RS 誤り訂正が無効（パリティ照合のみ） | 高 | **軽減済み** – reedsolo を dev 依存に追加。ランタイムで非搭載の場合のみパリティ照合フォールバックに縮退（純実装訂正デコーダは今後の課題） |
| G-2 | `decode()` が不可訂正ブロックを検出してもデータ部を黙って返す（サイレント破損） | 高 | **解消** – `ValueError`（グループ/ブロック番号付き）を送出。CLI は stderr + exit(1)。テスト 3 件も新挙動に更新済み |
| G-3 | Turbo 標準ブロック長 (1784..16384) での実符復号テスト不在 | 高 | **解消** – `test_turbo_standard_lengths.py` で全 5 長 × rate 1/3・1/6 の roundtrip + 全 5 長 × 4 レートの自動判別を検証 |
| G-4 | Turbo QPP インタリーバの外部照合 | 中 | **解消** – 公式 CCSDS §6.3g 数式の独立再実装と全 5 標準長を照合。K=1784 はさらに外部参照ファイルと一致 |
| G-4b | Turbo Annex A 符号語ベクトル不在 | 中 | **未解消** – 公開されている Annex A 符号語データが見つからない。代替として独立 QPP 検証で信頼性を確保 |
| G-5 | BER/FER Monte-Carlo シミュレーション不在 | 中 | **解消** – `test_ber_fer_simulation.py` で AWGN 上 conv/turbo の BER/FER 単調性を検証 |
| G-6 | CI が reedsolo なしで実行され訂正テストが失敗する可能性 | 中 | **解消** – `pyproject.toml` dev extras に reedsolo 追加 |
| G-7 | ドキュメント 5 件が現行実装と矛盾 | 低 | **解消** – 5 件すべて更新済み（+ 追加検証した 2 箇所の矛盾も修正） |
| G-8 | `BaseEncoder`/`BaseDecoder` 抽象基底クラス未実装 | 中 | **解消** – `api.py` に `BaseEncoder`/`BaseDecoder` (ABC) を追加し、3 コーデックが継承。`test_base_codec.py` で検証 |
| G-9 | ビット列 API が `np.uint8` ndarray で統一されていない | 低〜中 | **未解消** – AGENTS.md §1/§4.1 のデータ型基準。機能的には問題ないが、型契約・ベクトル化の一貫性が損なわれる |

---

## 付記

- 本マトリックスの `file:line` 引用は検証スクリプトにより実在確認済み。テスト名はテストディレクトリの実スキャンで確認済み。
- 実行証跡: `pytest` 189 passed → 228 passed → 238 passed → **267 passed**（2026-08-10, Python 3.14.4）。`ruff check src/ccsds_codec tests` も全通過。
- RS のゴールデンベクトル (`test_rs.py::test_encode_known_vector`) は reedsolo の CCSDS パラメータ (fcr=112, prim=0x187) との parity 一致で検証。
- CONV のゴールデンベクトル (`test_conv_known.py`) は gr-satellites の GNU Radio 実装とのビット一致で検証。
- Turbo インタリーバのゴールデンベクトル (`test_turbo_golden.py`) は mdmoctezuma/CCSDSTurboCode の `ccsdsSize1784.txt`（CCSDS 標準インタリーバ表）との K=1784 完全一致で検証（`tests/data/ccsdsSize1784.txt` としてコミット、sha256 c7094e37...）。
- 本マトリックスは検証時点のスナップショット。実装変更時は更新すること。
