# EXP-0023 — 并行探测优化

批量 Q/Y 协议 + 按轮爬坡；ELF 时序保持串行（进程启动噪声），Python 时序并行。对照组为 EXP-0022 的串行实现。

## 对照（同机同靶场）

| 题 | v024 串行 | v025 并行 |
| --- | --- | --- |
| T2 Python 时序 | 约 3.2 s | 约 1.3 s |
| R1 ELF 数字前缀 | 约 3.5 s | 约 2.0 s |
| T1 ELF 时序 | 约 11–16 s，偶发失败 | 约 11 s，阈值上限 + 重试后稳定 |
| E4 ELF 变换 | — | 约 1.5 s |

## 汇总

```json
{
  "positive_total": 4,
  "positive_solved": 4,
  "negative_total": 2,
  "negative_correctly_refused": 2,
  "total_probes": 2171,
  "mean_elapsed_ms": 3864.6
}
```

## 逐题

| 题 | 类型 | 接口 | 结果 | 路径 / 原因 | 探测数 |
| --- | --- | --- | --- | --- | --- |
| T1 | 正样本（ELF 每匹配一字节 sleep 6ms） | timing | ✓ 解出 | solve_timing_checker | 387 |
| T2 | 正样本（Python 每匹配一字节 sleep 2ms） | timing | ✓ 解出 | solve_timing_checker | 485 |
| R1 | 正样本（数字前缀校验器（v023 回归）） | check_num | ✓ 解出 | solve_prefix_checker | 594 |
| E4 | 正样本（ELF 变换程序（批量加速）） | enc | ✓ 解出 | probe_facts → classify_structure → invert_bytewise_affine → verify_plain | 221 |
| N1 | 负样本（恒定 sleep 5ms，与输入无关） | timing | ✓ 如实拒绝 | timing_signal_weak | 484 |
| N2 | 负样本（每字节 0.2ms，埋在进程噪声里） | none | ✓ 如实拒绝 | no_prefix_signal | 0 |