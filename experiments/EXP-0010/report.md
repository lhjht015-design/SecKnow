# EXP-0010 — 多成员压缩包

`unzip` 返回全部成员，规划器对每个成员分叉，按成员先验排序后用 beam 控预算。

## 汇总

```json
{
  "search": "beam",
  "beam_k": 32,
  "units": 8,
  "positive_total": 12,
  "positive_solved": 12,
  "negative_total": 2,
  "negative_correctly_failed": 2,
  "mean_expanded": 17.1,
  "mean_elapsed_ms": 14.1
}
```

## 逐题

| 题 | 题面 | 路径（尾部） | 展开 | 结果 |
| --- | --- | --- | --- | --- |
| Z1 | specific | unzip:secret.txt→read_text | 2 | ✓ |
| Z1 | abstract | unzip:secret.txt→read_text | 2 | ✓ |
| Z2 | specific | unzip:payload.zip→unzip:inside.txt→read_text | 23 | ✓ |
| Z2 | abstract | unzip:payload.zip→unzip:inside.txt→read_text | 23 | ✓ |
| Z3 | specific | unzip:image.png→png_text→b64_decode | 13 | ✓ |
| Z3 | abstract | unzip:image.png→png_text→b64_decode | 13 | ✓ |
| Z4 | specific | unzip:zzz_9f3a.dat→read_text | 50 | ✓ |
| Z4 | abstract | unzip:zzz_9f3a.dat→read_text | 50 | ✓ |
| Z5 | specific | unzip:flag.txt→read_text | 4 | ✓ |
| Z5 | abstract | unzip:flag.txt→read_text | 4 | ✓ |
| N1 | specific | — | 25 | ✓(正确失败) |
| N1 | abstract | — | 25 | ✓(正确失败) |
| N2 | specific | unzip:level1.zip→unzip:deep.txt→read_text | 3 | ✓ |
| N2 | abstract | unzip:level1.zip→unzip:deep.txt→read_text | 3 | ✓ |