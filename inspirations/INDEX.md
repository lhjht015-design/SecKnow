# inspirations/INDEX — 从顶刊/顶会借鉴的思路

> 每条形如 `INS-000N-<slug>.md`，记录：出处、核心思想、**迁移到本项目的具体方案**、可验证的预期。
> 只记录真正要用的，不堆砌。

| id | 来源 | 核心思想 | 迁移方案 | 状态 |
| --- | --- | --- | --- | --- |
| [INS-0001](INS-0001-sag2.md) | arXiv:2610.22588（2026-10） | SAG2：BM25 + 查询时多跳词扩展，建库零模型调用 | 用本地 reranker 替换在线 JEV，已实现为 v005 | **主路线，评测中** |
| [INS-0002](INS-0002-reap.md) | AAAI 2026（arXiv:2511.09966） | REAP：结构化子任务计划 + 事实提取，递归自纠正 | 需要 LLM，暂不实现；先借用"潜在线索"检查 | 暂缓 |

## 文献来源

- 本地文献库：`reference/Paper-Notes`（zhaoyang97/Paper-Notes，24,475 篇，覆盖
  AAAI/ACL/CVPR/ECCV/ICCV/ICLR/ICML/NeurIPS 2024–2026）。
  检索方式：按目录 + 文件名关键词，例如
  `find reference/Paper-Notes/docs -name "*.md" | grep -iE "multi_hop|retrieval_augmented|bm25"`
- 会议内的 `information_retrieval` 子目录是检索类论文的集中地。

## 选文原则

只收**能落到我们这个约束下**的思路：离线、尽量不依赖生成式大模型、能在现有
SQLite FTS5 + reranker 栈上验证。纯理论或需要在线 API 的，记录但不实现。
