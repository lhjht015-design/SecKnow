# EXP-0024 — 变换单元全部走批量 Y

四条反解路径都在编译后的 ELF 上验证：建表 / 仿射 / 置换 / GF(2)。
所有变换探测经 `call_many` 走 Y 批量（8 路并发、保序），`verify_plain` 的单次复核也走同一路径。

## 汇总

```json
{
  "positive_total": 4,
  "positive_solved": 4,
  "negative_total": 0,
  "negative_correctly_refused": 0,
  "total_probes": 2088,
  "mean_elapsed_ms": 3666.9
}
```

## 逐题

| 题 | 类型 | 接口 | 结果 | 路径 / 原因 | 探测数 |
| --- | --- | --- | --- | --- | --- |
| V1 | 正样本（ELF 逐字节 S 盒（建表路径）） | enc | ✓ 解出 | probe_facts → classify_structure → invert_bytewise_table → verify_plain | 1465 |
| V2 | 正样本（ELF 逐字节仿射（仿射路径）） | enc | ✓ 解出 | probe_facts → classify_structure → invert_bytewise_affine → verify_plain | 206 |
| V3 | 正样本（ELF 位置置换 + 常量（置换路径）） | enc | ✓ 解出 | probe_facts → classify_structure → invert_permutation → verify_plain | 145 |
| V4 | 正样本（ELF GF(2) 循环线性（稠密路径）） | enc | ✓ 解出 | probe_facts → classify_structure → invert_gf2 → verify_plain | 272 |