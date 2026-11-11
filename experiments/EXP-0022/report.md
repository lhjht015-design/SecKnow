# EXP-0022 — 时序侧信道的前缀爬坡

接口检测新增 timing；爬坡用单轮扫描 + 中位数确认 + 结束验证。

## 汇总

```json
{
  "positive_total": 3,
  "positive_solved": 3,
  "negative_total": 2,
  "negative_correctly_refused": 2,
  "total_probes": 1177,
  "mean_elapsed_ms": 4080.2
}
```

## 逐题

| 题 | 类型 | 接口 | 结果 | 路径 / 原因 | 探测数 |
| --- | --- | --- | --- | --- | --- |
| T1 | 正样本（ELF 每匹配一字节 sleep 6ms） | timing | ✓ 解出 | solve_timing_checker | 384 |
| T2 | 正样本（Python 每匹配一字节 sleep 2ms） | timing | ✓ 解出 | solve_timing_checker | 374 |
| R1 | 正样本（数字前缀校验器（v023 回归）） | check_num | ✓ 解出 | solve_prefix_checker | 419 |
| N1 | 负样本（恒定 sleep 5ms，与输入无关） | none | ✓ 如实拒绝 | no_prefix_signal | 0 |
| N2 | 负样本（每字节 0.2ms，埋在进程噪声里） | none | ✓ 如实拒绝 | no_prefix_signal | 0 |