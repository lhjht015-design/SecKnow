# EXP-0021 — ELF 校验器的前缀接口

接口检测归一成 enc / check_num / check_exit / none；前缀信号交给逐字节爬坡。

## 汇总

```json
{
  "positive_total": 4,
  "positive_solved": 4,
  "negative_total": 2,
  "negative_correctly_refused": 2,
  "total_probes": 1100,
  "mean_elapsed_ms": 2339.3
}
```

## 逐题

| 题 | 类型 | 接口 | 结果 | 路径 / 原因 | 探测数 |
| --- | --- | --- | --- | --- | --- |
| E1 | 正样本（stdout 输出前缀长度整数） | check_num | ✓ 解出 | solve_prefix_checker | 322 |
| E2 | 正样本（输出 prefix=N） | check_num | ✓ 解出 | solve_prefix_checker | 318 |
| E3 | 正样本（前缀长度编码在退出码） | check_exit | ✓ 解出 | solve_prefix_checker | 337 |
| E4 | 正样本（ELF 变换程序（回归）） | enc | ✓ 解出 | probe_facts → classify_structure → invert_bytewise_affine → verify_plain | 123 |
| N5 | 负样本（只回 wrong 的校验器） | none | ✓ 如实拒绝 | no_prefix_signal | 0 |
| N6 | 负样本（输出恒为 0（不随输入变化）） | none | ✓ 如实拒绝 | no_prefix_signal | 0 |