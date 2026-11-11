# EXP-0014 — ret2libc

两阶段：`puts(puts@got)` 泄漏 → 算 libc 基址 → `system("/bin/sh")`。

## 汇总

```json
{
  "positive_total": 1,
  "positive_solved": 1,
  "negative_total": 2,
  "negative_correctly_refused": 2,
  "mean_elapsed_ms": 410.3
}
```

## 逐题

| 题 | 类型 | 结果 | 原因 | 耗时 |
| --- | --- | --- | --- | --- |
| L1 | 正样本 | ✓ 解出 | — | 1040.8ms |
| N1 | 负样本 | ✓ 如实拒绝 | libc_mismatch | 187.3ms |
| N2 | 负样本 | ✓ 如实拒绝 | plt_stub_missing | 2.8ms |