# EXP-0012 — pwn 单元（真实 ELF + 真实执行）

四个单元：`elf_info` → `find_gadgets` → `probe_offset` → `build_ret2win`，
最后在 WSL 里真实执行二进制验证。

## 汇总

```json
{
  "positive_total": 2,
  "positive_solved": 2,
  "negative_total": 2,
  "negative_correctly_refused": 2,
  "mean_elapsed_ms": 475.2
}
```

## 逐题

| 题 | 类型 | 结果 | 原因 | 用到的单元 | 耗时 |
| --- | --- | --- | --- | --- | --- |
| T1 | 正样本 | ✓ 解出 | — | elf_info → find_gadgets → probe_offset → build_ret2win → execute | 1320.9ms |
| T2 | 正样本 | ✓ 解出 | — | elf_info → find_gadgets → probe_offset → build_ret2win → execute | 572.1ms |
| N1 | 负样本 | ✓ 正确拒绝 | canary_present | elf_info → protection_gate | 4.0ms |
| N2 | 负样本 | ✓ 正确拒绝 | pie_enabled | elf_info → protection_gate | 3.9ms |