# EXP-0019 — 8 个黑盒单元并入文本召回库

同一份 UNITS / recall / 规划器同时处理文本题与黑盒脚本题。

## 汇总

```json
{
  "units": 14,
  "positive_total": 11,
  "positive_solved": 11,
  "negative_total": 4,
  "negative_correctly_refused": 4,
  "total_probes": 16894
}
```

## 逐题

| 题 | 类型 | 结果 | 规划路径 / 原因 | 召回 | 探测数 |
| --- | --- | --- | --- | --- | --- |
| T1 | 正样本（RSA 小公钥指数） | ✓ 解出 | rsa_small_e | rsa_small_e | 0 |
| T2 | 正样本（多层 Base64） | ✓ 解出 | base64_chain | base64_chain | 0 |
| T3 | 正样本（单字节 XOR） | ✓ 解出 | xor_single_byte | xor_single_byte, crc32_brute | 0 |
| T4 | 正样本（CRC32 爆破） | ✓ 解出 | crc32_brute | crc32_brute | 0 |
| T5 | 正样本（凯撒密码） | ✓ 解出 | caesar_brute | caesar_brute | 0 |
| N4 | 负样本（RSA e=65537（文本题负样本）） | ✓ 如实拒绝 | not_perfect_power | rsa_small_e | 0 |
| P1 | 正样本（逐字节 S 盒） | ✓ 解出 | probe_facts → classify_structure → invert_bytewise_table → verify_plain | invert_bytewise_table, invert_bytewise_affine | 6278 |
| P2 | 正样本（逐字节仿射） | ✓ 解出 | probe_facts → classify_structure → invert_bytewise_affine → verify_plain | invert_bytewise_affine, invert_bytewise_table | 124 |
| P3 | 正样本（GF(2) 矩阵） | ✓ 解出 | probe_facts → classify_structure → invert_gf2 → verify_plain | invert_gf2 | 220 |
| P4 | 正样本（三轮 Feistel） | ✓ 解出 | probe_facts → classify_structure → invert_gf2 → verify_plain | invert_gf2 | 239 |
| P5 | 正样本（前缀校验器） | ✓ 解出 | solve_prefix_checker | solve_prefix_checker | 375 |
| P6 | 正样本（黑盒 S 盒 + Base64 组合） | ✓ 解出 | probe_facts → classify_structure → invert_bytewise_table → verify_plain → bytes_to_text → base64_chain | invert_bytewise_table, base64_chain, invert_bytewise_affine | 9410 |
| N1 | 负样本（sha256） | ✓ 如实拒绝 | not_length_preserving |  | 13 |
| N2 | 负样本（非线性混合） | ✓ 如实拒绝 | structure_unknown | invert_gf2 | 232 |
| N3 | 负样本（随机掩码） | ✓ 如实拒绝 | nondeterministic |  | 3 |