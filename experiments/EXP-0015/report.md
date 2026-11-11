# EXP-0015 — ret2win 复测（stripped / -O2 / 改名）

验证符号剥离、优化与函数改名后，后门还能不能被定位；
所有候选都靠真实执行验证，看到一个 `flag{...}` 不算成功。

## 汇总

```json
{
  "positive_total": 4,
  "positive_solved": 4,
  "negative_total": 2,
  "negative_correctly_refused": 2,
  "mean_elapsed_ms": 623.7
}
```

## 逐题

| 题 | 靶场 | 结果 | 找到方式 / 原因 | 尝试次数 | 耗时 |
| --- | --- | --- | --- | --- | --- |
| R0 | vuln_o0 | ✓ 解出 | 正样本：symbol:win | 1 | 672.1ms |
| R1 | vuln_o2 | ✓ 解出 | 正样本：symbol:win | 1 | 740.4ms |
| R2 | vuln_o2s | ✓ 解出 | 正样本：string_ref:/tmp/flag.txt | 1 | 726.1ms |
| R3 | vuln_renamed_o2 | ✓ 解出 | 正样本：string_ref:/tmp/flag.txt | 1 | 752.1ms |
| N1 | vuln_nowin_o2s | ✓ 如实拒绝 | 负样本：win_not_found | — | 3.4ms |
| N2 | vuln_decoy_o2s | ✓ 如实拒绝 | 负样本：execute_failed | 2 | 847.9ms |