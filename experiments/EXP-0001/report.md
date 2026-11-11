# EXP-0001 — 离线推理解题可行性判决性实验

生成时间：2026-09-21 15:39:36
模型：qwen3:8b

## 判据说明

- `A_solve_rate`：题目**真的被解出**的比例。`solved_by_exec` 表示模型写的代码被真实执行并打印出正确 flag。
- `A_false_positive_rate`：模型给出了一个 flag 但**是错的**。这是「假能解」的直接度量。
- `B_ungrounded_citation_rate`：回答里引用了不存在的证据编号的比例（幻觉引用）。
- `C_false_confidence_rate`：对知识库中根本没有答案的问题**仍然自信作答**的比例。这正是浙密杯模式的量化指标。

## 汇总

### qwen3:8b

| 指标 | 值 |
| --- | --- |
| total | 9 |
| A_tasks | 9 |
| A_solved_by_exec | 1 |
| A_solved_inline_only | 0 |
| A_solve_rate | 0.111 |
| A_exec_rate | 0.111 |
| A_false_positive_rate | 0.667 |
| A_mean_latency_s | 13.1 |
| A_mean_attempts | 1.0 |
| A_solved_on_first_try | 1 |

## 逐题结果（Regime A）

| 题号 | 类别 | 模型 | 判定 | 期望 flag | 模型声称 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | encoding | qwen3:8b | failed | `flag{b4se64_ch41n_1s_e4sy}` | `—` | 3.43s |
| A2 | crypto | qwen3:8b | failed | `flag{x0r_w1th_s1ngle_byte}` | `flag{flag{x0r_w1th_s1ngle_byte}` | 4.51s |
| A3 | crypto | qwen3:8b | solved_by_exec | `flag{caesar_shifts_by_13}` | `flag{caesar_shifts_by_13}` | 4.55s |
| A4 | crypto | qwen3:8b | failed | `flag{rsa_small_e_no_padding}` | `flag{{{m}` | 47.72s |
| A5 | encoding | qwen3:8b | failed | `flag{b1n4ry_t0_4sc11}` | `flag{011001100110110001100001011001110111101101100010001100010110111000110100011100100111100101011111011101000011000001011111001101000111001101100011001100010011000101111101}` | 8.74s |
| A6 | reverse | qwen3:8b | failed | `flag{rev3rs1ng_l0g1c}` | `—` | 4.9s |
| A7 | misc | qwen3:8b | failed | `flag{ctfz}` | `flag{{{s}` | 4.97s |
| A8 | crypto | qwen3:8b | failed | `flag{fermat_factors_close_primes}` | `flag{{{flag}` | 30.34s |
| A9 | crypto | qwen3:8b | failed | `flag{repeating_key_xor_needs_crib}` | `flag{{{zlib.decompress(decrypted).decode()}` | 8.87s |
