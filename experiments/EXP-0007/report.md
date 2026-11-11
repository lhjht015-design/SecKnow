# EXP-0007 — 多步组合验证

单元按「输入/输出类型」串成流水线，规划器做类型化 BFS，每一步真实执行。

## 汇总

```json
{
  "positive_total": 8,
  "positive_solved": 8,
  "negative_total": 2,
  "negative_correctly_failed": 2,
  "by_depth": {
    "depth_1": {
      "solved": 2,
      "total": 2
    },
    "depth_2": {
      "solved": 2,
      "total": 2
    },
    "depth_3": {
      "solved": 2,
      "total": 2
    },
    "depth_4": {
      "solved": 2,
      "total": 2
    }
  },
  "mean_expanded": 216.4,
  "mean_elapsed_ms": 7.7
}
```

## 逐题

| 题 | 题面 | 期望步数 | 实际路径 | 展开节点 | 耗时 | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | specific | 1 | xor_brute | 4 | 0.2ms | ✓ |
| S1 | abstract | 1 | xor_brute | 4 | 0.1ms | ✓ |
| S2 | specific | 2 | b64_decode→xor_brute | 23 | 0.5ms | ✓ |
| S2 | abstract | 2 | b64_decode→xor_brute | 23 | 0.4ms | ✓ |
| S3 | specific | 3 | b64_decode→hex_decode→caesar_brute | 158 | 3.2ms | ✓ |
| S3 | abstract | 3 | b64_decode→hex_decode→caesar_brute | 158 | 3.1ms | ✓ |
| S4 | specific | 4 | b64_decode→hex_decode→b64_decode→xor_brute | 314 | 14.9ms | ✓ |
| S4 | abstract | 4 | b64_decode→hex_decode→b64_decode→xor_brute | 314 | 15.0ms | ✓ |
| N1 | specific | 2 | — | 583 | 20.4ms | ✓(正确失败) |
| N1 | abstract | 2 | — | 583 | 19.6ms | ✓(正确失败) |