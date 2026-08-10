# CCSDS 準拠検証マトリックス

**対象規格**: CCSDS 131.0-B-4 (TM Synchronization and Channel Coding)
**対象実装**: `ccsds-codec` (Python 3.11+)
**検証日**: 2026-08-10
**検証環境**: Python 3.14.4 / numpy + numba / reedsolo (import 可) / pytest 189 passed

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
| RS-05 | §4.2 | 誤り訂正能力 t=16 | `core/reed_solomon.py:154` (`reedsolo.RSCodec(32, nsize=255, fcr=112, prim=0x187)`) | `test_rs_properties.py::test_error_correction_with_reedsolo`, `test_edge_cases.py::test_rs_decode_with_correctable_errors` | ⚠️ 部分的 | **reedsolo 非搭載時は訂正なし**（パリティ照合のみのフォールバック）。リスク項目 G-1 |
| RS-06 | §4.2 | 不可訂正エラーの検出 | `core/reed_solomon.py:127-140` (`_fallback_decode_block`), `:164-207` (`decode`) | `test_rs_internal.py::test_internal_decode_too_many_errors`, `test_edge_cases.py::test_rs_decode_exceeds_error_capacity`, `test_rs_extended.py::test_decode_without_errors_fallback` | ⚠️ 部分的 | **`decode()` は不可訂正ブロックを検出するとデータ部を黙って返す**（`block[:RS_K]` フォールバック、行 204）。リスク項目 G-2 |
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
| TURBO-03 | §3.1.1 | 情報ブロック長 1784/3568/7136/8920/16384 | `core/turbo.py:100` (`STANDARD_K`) | `test_turbo_properties.py::test_rate_autodetect_is_unique_and_correct` | ⚠️ 部分的 | **レート自動判別のみ検証。標準長での実符復号テスト不在**（全テスト K≤200）。リスク項目 G-3 |
| TURBO-04 | §3.2.3 | 終端（K−1=4 テール、状態 0 リセット） | `core/turbo.py:76` (`TAIL = 4`) | `test_turbo_properties.py::test_stream_length_formula` | ✅ 準拠 | 長さ公式で間接検証 |
| TURBO-05 | §6.3g | QPP インタリーバ（Quadratic-Permutation） | `core/interleaver.py` (`ccsds_perm` / `ccsds_interleaver` / `ccsds_deinterleaver`) | `test_turbo_extended.py::test_interleaver_is_bijective_and_inverse`, `test_turbo_properties.py::test_interleaver_is_permutation`, `test_turbo_properties.py::test_deinterleaver_is_inverse` | ✅ 準拠 | 全単射・逆元を検証。**外部参照 (gr-ccsds-1/SatDump) との一致テスト不在**。リスク項目 G-4 |
| TURBO-06 | §3.4 (Annex) | 反復 Log-MAP / Max-Log-MAP 復号 | `core/turbo.py:215` (`_build_trellis`), `:256` (`_bcjr_kernel`), `_turbo_decode_core` | `test_turbo.py::test_unpunctured_roundtrip`, `test_turbo.py::test_punctured_roundtrip`, `test_turbo_extended.py::test_decode_consistency_across_iterations` | ✅ 準拠 | |
| TURBO-07 | 数値安定性 | 対数領域でアンダーフロー防止 | `core/turbo.py:256` (`_bcjr_kernel`) | `test_turbo_rate16.py::test_bcjr_kernel_channel_matrix_shapes`, `test_turbo_rate16.py::test_bcjr_kernel_clean_channel_decodes_zeros` | ✅ 準拠 | 有限 LLR・チャネル行列形状を検証 |
| TURBO-08 | §3.1.1 | ブロック長からのレート自動判別 | `core/turbo.py` (`_detect_rate_k`) | `test_turbo_properties.py::test_rate_autodetect_is_unique_and_correct`, `test_turbo_properties.py::test_rate_autodetect_rejects_unknown_length` | ✅ 準拠 | |
| TURBO-09 | Annex A | 付録の基準テストベクトル | なし | テストなし | ❓ 未検証 | 付録ベクトルとの照合テスト不在。リスク項目 G-4 |
| TURBO-10 | §3.4 | 最大反復回数（規格例: 10 回） | デフォルト 5 回、上限未設定 | `test_turbo_extended.py::test_decode_consistency_across_iterations` | ⚠️ 部分的 | 反復回数は設定可能だが上限強制なし |

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

## 6. 性能・品質目標

| ID | 要求 | 要求内容 | 実装箇所 | 検証テスト | ステータス | 備考・リスク |
|---|---|---|---|---|---|---|
| PERF-01 | AGENTS.md §3 (Tester) | AWGN 上 Monte-Carlo BER/FER | なし | テストなし | ✗ 非準拠 | **BER/FER シミュレーション実装なし**。README の「golden vectors, BER/FER sims」記述と乖離。リスク項目 G-5 |
| PERF-02 | AGENTS.md §4.2 | 計算コアの numba JIT | `core/convolutional.py` (Viterbi カーネル), `core/turbo.py:256` (`_bcjr_kernel`) | `test_turbo_perf.py::test_decode_performance` | ⚠️ 部分的 | 性能テストは 1 件のみ。numba 非搭載環境での動作は未検証 |

---

## 7. CI・ドキュメント整合性

| ID | 対象 | 問題内容 | ステータス | 備考 |
|---|---|---|---|---|
| CI-01 | `.github/workflows/ci.yml` | `pip install .[dev]` に reedsolo が含まれない → CI で `test_rs_decode_with_correctable_errors` 等が失敗する可能性 | ⚠️ リスク | ローカルでは reedsolo が import 可能なため 189 passed だが、クリーン環境の CI はフォールバック動作になり訂正能力テストが落ちる。リスク項目 G-6 |
| DOC-01 | `docs/ccsds_spec.md` | CONV 生成多項式が `G1=121₈`（TC 方式）表記で、実装 `171₈/133₈` と矛盾 | ✗ 非準拠 | 実装は CCSDS 131.0-B-4 準拠で正しい。ドキュメント側の修正が必要 |
| DOC-02 | `docs/COMPATIBILITY.md` | Turbo が「簡易デモ・非互換」と記載 → 実装は正式 RSC/QPP/Log-MAP に更新済み | ✗ 非準拠 | 陳腐化。リスク項目 G-7 |
| DOC-03 | `docs/COMPATIBILITY_TURBO.md` | f1=17/f2=31 の旧インタリーバ記述が残存 | ✗ 非準拠 | 現行 QPP 実装と矛盾 |
| DOC-04 | `docs/COMPATIBILITY_CLTU.md` | 存在しない `ccsds_codec.cltu` モジュールへの参照 | ✗ 非準拠 | モジュールは存在せず、参照先が無効 |
| DOC-05 | `docs/architecture.md` | `src/ccsds/` レイアウト記述が現行 `src/ccsds_codec/` と不一致 | ✗ 非準拠 | 陳腐化 |

---

## サマリー

| カテゴリ | ✅ 準拠 | ⚠️ 部分的 | ✗ 非準拠 | ❓ 未検証 |
|---|---|---|---|---|
| Reed-Solomon | 4 | 2 | — | 1 |
| Convolutional | 7 | — | — | — |
| Turbo | 6 | 2 | — | 1 |
| Randomizer | 4 | — | — | — |
| 共通規約 | 2 | — | — | — |
| 性能目標 | — | 1 | 1 | — |
| CI・ドキュメント | — | 1 | 5 | — |
| **合計** | **23** | **6** | **6** | **2** |

---

## 主要リスク（G-1..G-7）と推奨アクション

| ID | リスク | 深刻度 | 推奨アクション |
|---|---|---|---|
| G-1 | reedsolo 非搭載環境で RS 誤り訂正が無効（パリティ照合のみ） | 高 | reedsolo を必須依存化するか、純実装の訂正デコーダを追加 |
| G-2 | `decode()` が不可訂正ブロックを検出してもデータ部を黙って返す（サイレント破損） | 高 | エラー送出 or 訂正失敗フラグ付き戻り値に変更（API 互換に注意） |
| G-3 | Turbo 標準ブロック長 (1784..16384) での実符復号テスト不在 | 高 | 標準長での roundtrip / 誤り訂正テストを追加 |
| G-4 | Turbo に外部参照ゴールデンベクトルテスト不在 | 中 | gr-ccsds-1 / SatDump のテストベクトルで照合 |
| G-5 | BER/FER Monte-Carlo シミュレーション不在 | 中 | AWGN チャネルでの BER/FER テスト追加 |
| G-6 | CI が reedsolo なしで実行され訂正テストが失敗する可能性 | 中 | CI に reedsolo を追加するか、非搭載時 skip |
| G-7 | ドキュメント 5 件が現行実装と矛盾 | 低 | ドキュメント更新 |

---

## 付記

- 本マトリックスの `file:line` 引用は検証スクリプトにより実在確認済み。テスト名はテストディレクトリの実スキャンで確認済み。
- 実行証跡: `pytest` 189 passed (2026-08-10, Python 3.14.4)。
- RS のゴールデンベクトル (`test_rs.py::test_encode_known_vector`) は reedsolo の CCSDS パラメータ (fcr=112, prim=0x187) との parity 一致で検証。
- CONV のゴールデンベクトル (`test_conv_known.py`) は gr-satellites の GNU Radio 実装とのビット一致で検証。
- 本マトリックスは検証時点のスナップショット。実装変更時は更新すること。
