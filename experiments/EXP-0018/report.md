# EXP-0018 — 黑盒反解策略接入契约单元库

同一套类型化规划器：按状态类型选单元、按 guard 拒绝不适用、最后统一做执行验证。

## 汇总

```json
{
  "units": 8,
  "positive_total": 5,
  "positive_solved": 5,
  "negative_total": 3,
  "negative_correctly_refused": 3,
  "total_probes": 7484
}
```

## 逐题

| 题 | 类型 | 结果 | 规划路径 / 原因 | 探测数 |
| --- | --- | --- | --- | --- |
| P1 | 正样本（逐字节 S 盒） | ✓ 解出 | probe_facts → classify_structure → invert_bytewise_table → verify_plain | 6278 |
| P2 | 正样本（逐字节仿射） | ✓ 解出 | probe_facts → classify_structure → invert_bytewise_affine → verify_plain | 124 |
| P3 | 正样本（GF(2) 随机可逆矩阵） | ✓ 解出 | probe_facts → classify_structure → invert_gf2 → verify_plain | 220 |
| P4 | 正样本（三轮 Feistel） | ✓ 解出 | probe_facts → classify_structure → invert_gf2 → verify_plain | 239 |
| P5 | 正样本（前缀校验器） | ✓ 解出 | solve_prefix_checker | 375 |
| N1 | 负样本（sha256） | ✓ 如实拒绝 | not_length_preserving | 13 |
| N2 | 负样本（非线性 32 位混合） | ✓ 如实拒绝 | structure_unknown | 232 |
| N3 | 负样本（随机掩码） | ✓ 如实拒绝 | nondeterministic | 3 |