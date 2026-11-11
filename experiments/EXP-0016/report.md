# EXP-0016 — canary + PIE + ret2libc

同一进程内：`%p` 泄漏 canary 与 PIE 基址 → `%s` 泄漏 puts → ROP 调 system。

## 汇总

```json
{
  "positive_total": 1,
  "positive_solved": 1,
  "negative_total": 2,
  "negative_correctly_refused": 2,
  "mean_elapsed_ms": 496.1
}
```

## 逐题

| 题 | 类型 | 结果 | 原因 | 耗时 |
| --- | --- | --- | --- | --- |
| T1 | 正样本 | ✓ 解出 | — | 1188.4ms |
| N1 | 负样本 | ✓ 如实拒绝 | libc_mismatch | 146.6ms |
| N2 | 负样本 | ✓ 如实拒绝 | leak_failed | 153.4ms |