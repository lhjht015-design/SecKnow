# EXP-0013 — canary / PIE 绕过

核心约束：泄漏与利用必须在**同一进程**内完成（ASLR 每次运行都变）。

## 汇总

```json
{
  "positive_total": 3,
  "positive_solved": 3,
  "negative_total": 1,
  "negative_correctly_failed": 1,
  "mean_elapsed_ms": 1779.1
}
```

## 逐题

| 题 | 类型 | 结果 | 原因 | 耗时 |
| --- | --- | --- | --- | --- |
| C1 | 正样本 | ✓ 解出 | — | 506.3ms |
| P1 | 正样本 | ✓ 解出 | — | 247.1ms |
| B1 | 正样本 | ✓ 解出 | — | 247.7ms |
| N1 | 负样本 | ✓ 如实失败 | leak_stage_not_found | 6115.2ms |