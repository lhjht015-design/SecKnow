# EXP-0004 — SAG2 式离线多跳检索

生成时间：2026-11-11 10:45:28
参数：样本 120，Top-K 8，扩展轮数 2，候选 40，判官保留 6，每轮扩展词 5

## 做法

```
BM25(FTS5) 取候选  →  bge-reranker 当判官  →  jieba 从选中段落抽新词  →  再检索
```

JEV 在论文里是在线判官 API；这里用本地 cross-encoder 替换，**全程离线、不调用大模型**。

## 汇总

```json
{
  "queries": 120,
  "top_k": 8,
  "rounds": 2,
  "tag_recall_at_k": 0.994,
  "tag_recall_excl_self": 0.1751,
  "self_leak_rate": 1.0,
  "similar_wp_rate": 0.1,
  "similar_wp_rate_conditional": 0.3,
  "similar_wp_subset_size": 40,
  "vuln_type_hit_rate": 0.1833,
  "mean_expanded_terms": 10.0,
  "elapsed_s": 662.3,
  "per_query_s": 5.52,
  "top_k_composition": {
    "meta_WP": 0.96,
    "book_PDF": 0.027,
    "other": 0.013
  },
  "most_missed_tags": [
    [
      "2236416",
      1
    ],
    [
      "3404319313",
      1
    ],
    [
      "\"Wanna return?\" A",
      1
    ],
    [
      "\"once again?\" A*0x100 二次溢出",
      1
    ],
    [
      "LLL格密码",
      1
    ],
    [
      "fmtstr泄露canary+elfbase",
      1
    ]
  ]
}
```

## 与 v004（单跳 + 向量）对照

| 指标 | v004（单跳） | v005（多跳） |
| --- | --- | --- |
| tag_recall@K | 0.5925 | 0.994 |
| **tag_recall@K（排除自身文件）** | 0.5925* | **0.1751** |
| similar_wp_rate_conditional | 0.20 | 0.3 |
| self_leak_rate | 0.825 | 1.0 |

> \* v004 未单独输出排除自身文件的口径，其 0.5925 是含自身文件的原始值。
> 对比时请以 v005 的 `tag_recall_excl_self` 为准，或对 v004 重跑同一口径。