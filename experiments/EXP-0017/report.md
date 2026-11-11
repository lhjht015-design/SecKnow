# EXP-0017 — 黑盒刻画：不读逻辑也能反解

不读源码，只通过执行探测判定结构类别；判不出就如实拒绝。

## 汇总

```json
{
  "positive_total": 5,
  "positive_solved": 5,
  "negative_total": 3,
  "negative_correctly_refused": 3,
  "mean_elapsed_ms": 22.9,
  "total_probes": 7484
}
```

## 逐题

| 题 | 类型 | 结果 | 策略 / 原因 | 探测数 | 耗时 |
| --- | --- | --- | --- | --- | --- |
| P1 | 正样本（逐字节 S 盒） | ✓ 解出 | bytewise_table | 6278 | 40.2ms |
| P2 | 正样本（逐字节仿射） | ✓ 解出 | bytewise_affine | 124 | 1.6ms |
| P3 | 正样本（GF(2) 随机可逆矩阵） | ✓ 解出 | gf2_linear | 220 | 12.8ms |
| P4 | 正样本（三轮 Feistel（轮函数线性）） | ✓ 解出 | gf2_linear | 239 | 6.4ms |
| P5 | 正样本（前缀校验器） | ✓ 解出 | prefix_oracle | 375 | 120.3ms |
| N1 | 负样本（sha256（长度不保持）） | ✓ 如实拒绝 | not_length_preserving | 13 | 0.2ms |
| N2 | 负样本（非线性 32 位混合） | ✓ 如实拒绝 | structure_unknown | 232 | 1.9ms |
| N3 | 负样本（每次换随机掩码） | ✓ 如实拒绝 | nondeterministic | 3 | 0.1ms |