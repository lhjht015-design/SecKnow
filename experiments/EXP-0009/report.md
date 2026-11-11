# EXP-0009 — 文件型载荷

规划器状态新增 `FileRef`，按魔数识别 zip/png/pcap/elf，再由对应单元处理。

## 汇总

```json
{
  "search": "bfs",
  "units": 10,
  "positive_total": 12,
  "positive_solved": 12,
  "negative_total": 4,
  "negative_correctly_failed": 4,
  "mean_expanded": 2.9,
  "mean_elapsed_ms": 1.1
}
```

## 逐题

| 题 | 题面 | 实际路径 | 展开 | 结果 |
| --- | --- | --- | --- | --- |
| F1 | specific | read_text | 1 | ✓ |
| F1 | abstract | read_text | 1 | ✓ |
| F2 | specific | unzip→read_text | 2 | ✓ |
| F2 | abstract | unzip→read_text | 2 | ✓ |
| F3 | specific | unzip→png_text→b64_decode | 3 | ✓ |
| F3 | abstract | unzip→png_text→b64_decode | 3 | ✓ |
| F4 | specific | unzip→unzip→read_text | 3 | ✓ |
| F4 | abstract | unzip→unzip→read_text | 3 | ✓ |
| F5 | specific | pcap_strings | 1 | ✓ |
| F5 | abstract | pcap_strings | 1 | ✓ |
| F6 | specific | elf_strings | 1 | ✓ |
| F6 | abstract | elf_strings | 1 | ✓ |
| N1 | specific | — | 8 | ✓(正确失败) |
| N1 | abstract | — | 8 | ✓(正确失败) |
| N2 | specific | — | 4 | ✓(正确失败) |
| N2 | abstract | — | 4 | ✓(正确失败) |