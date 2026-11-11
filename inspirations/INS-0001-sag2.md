```yaml
id: INS-0001
prev: none
date: 2026-11-11
source: 论文 / arXiv
paper: "SAG2: Embedding-Free Multi-Hop Retrieval with BM25 and JEV"
venue: arXiv:2610.22588 (2026-10)
link: https://arxiv.org/abs/2610.22588
topic: 查询时多跳检索，embedding-free
```

## 核心思想

把「语义证据的选择」从**建库时**挪到**查询时**：

```
BM25(FTS5) 取候选 → JEV 判断哪些段落真的有用 → 被选中的段落提供新词 → 下一轮检索
```

默认约 2 轮扩展。建库**零模型调用**：11,656 段只要 2.36 秒、25.87 MiB 的 SQLite 库。
（对比 SAG v1 的 4.57 小时 / 2826 万 tokens，约 7000 倍加速。）

## 为什么这条对我们特别重要

1. **和我们现有栈完全同构**：我们已经是 SQLite FTS5 + BM25 + jieba 分词。
   迁移成本极低，不需要向量库、不需要 Docker。
2. **和我们的实测结论一致**：EXP-0003 显示 Top-K 里 83% 的命中来自 meta WP，
   书籍 PDF 只占 8.6%；向量通道开/关的差异很小。
   也就是说胜负手在「词法命中的精度 + 多跳补全」，不在向量空间的重组。
3. **建库成本低三个数量级**：我们现在全量 BGE 编码 16k 块要 6 分钟 GPU，
   SAG2 的路线是秒级、纯 CPU。

## 关键陷阱：JEV 本身不是离线的

论文里的 JEV 是 TypeSafe 的**在线判官 API**（`api.typesafe.ai/v1/systemone`，
model `jev-latest`），备选是 Liquid AI 的 D1-ultra。
**SAG2 的架构可以离线，但 JEV 按发表形态不能。**

迁移时必须换掉 JEV，候选：

| 替代方案 | 离线 | 是否大模型 | 说明 |
| --- | --- | --- | --- |
| `bge-reranker-v2-m3`（本方案） | ✅ | ❌（cross-encoder） | 已在本地 GPU，直接当判官用 |
| 本地小 LLM（qwen3:4b/8b） | ✅ | ✅ | 更接近论文，但慢且违背"不用大模型"的约束 |
| Liquid AI D1-ultra | ✅（若本地可跑） | ✅ 小模型 | 论文给的备选，需确认权重可得性 |

## 迁移到本项目的具体方案

已实现为 `versions/v005_sag2_multihop.src`：
- 候选：`MetadataStore.search_fts`（BM25，jieba 分词）
- 判官：`BgeReranker.rerank(query, passages)`
- 扩展词：`jieba.analyse.extract_tags` 从判官选中的段落里抽，过滤停用词与原查询已有词
- 融合：RRF 合并多轮结果
- **全程离线、零联网、不调用生成式大模型**

## 可验证的预期

在同 120 题、同真值口径下对比 EXP-0003 的单跳基线：

- `tag_recall@8` 应高于 0.593
- `similar_wp_rate_conditional` 应高于 0.20
- 单查询耗时上升不超过 3 倍

若达不到，说明我们的语料不是"多跳型"的（CTF 题面往往已经点明技术方向，
不需要第二跳补词），此时应当放弃多跳、把预算花在语料覆盖上。
