# EXP-0008 — 混合场景：先解码再走 RSA

流水线中段从「字符串」切换为「结构化参数」（int_fields）。

## 汇总

```json
{
  "search": "bfs",
  "units": 14,
  "positive_total": 8,
  "positive_solved": 6,
  "negative_total": 4,
  "negative_correctly_failed": 4,
  "mean_expanded": 329.2,
  "mean_elapsed_ms": 33.1
}
```

## 逐题

| 题 | 题面 | 期望步数 | 实际路径 | 展开 | 结果 |
| --- | --- | --- | --- | --- | --- |
| M1 | specific | 3 | b64_decode→parse_int_fields→rsa_small_e | 74 | ✓ |
| M1 | abstract | 3 | b64_decode→parse_int_fields→rsa_small_e | 74 | ✓ |
| M2 | specific | 3 | hex_decode→parse_int_fields→rsa_small_e | 74 | ✓ |
| M2 | abstract | 3 | hex_decode→parse_int_fields→rsa_small_e | 74 | ✓ |
| M3 | specific | 5 | — | 600 | ✗ |
| M3 | abstract | 5 | — | 600 | ✗ |
| M4 | specific | 2 | parse_int_fields→rsa_small_e | 27 | ✓ |
| M4 | abstract | 2 | parse_int_fields→rsa_small_e | 27 | ✓ |
| N1 | specific | 2 | — | 600 | ✓(正确失败) |
| N1 | abstract | 2 | — | 600 | ✓(正确失败) |
| N2 | specific | 2 | — | 600 | ✓(正确失败) |
| N2 | abstract | 2 | — | 600 | ✓(正确失败) |