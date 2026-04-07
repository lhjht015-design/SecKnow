## Phase 1 边界

已完成：

- 统一数据模型：`ChunkRecord / SearchHit / BaselineBundle / ExportManifest`
- 统一服务契约：`ensure_zone / upsert / search / hybrid_search / delete / export_zone / get_baseline`
- 在线后端：`QdrantVectorStore`
- 离线后端：`FaissSqliteVectorStore`
- dense + sparse 融合：`HybridRetriever`

暂不做：

- IVF-PQ 参数优化
- 完整 REST API
- 前端联调
- 真实召回质量调优

# Phase 1 设计说明（4.3）

## 1. 目标

第一阶段只做一件事：把 4.3 模块从 demo 脚本升级为可持续演进的检索基础设施层。  
即：**先定边界和契约，再做性能优化。**

## 2. 4.3 对外契约

统一通过 `VectorInfrastructureService` 暴露：

- `ensure_zone(zone_id, dim)`
- `upsert(zone_id, records)`
- `search(zone_id, query_vec, top_k, filters)`
- `hybrid_search(zone_id, query, query_vec, top_k, filters)`
- `delete(zone_id, chunk_ids)`
- `export_zone(zone_id, target_dir)`
- `get_baseline(zone_id)`

## 3. 模块分层

- 领域模型层：`src/secknow/vector_store/models.py`
- 抽象接口层：`src/secknow/vector_store/stores/base.py`
- 后端实现层：
  - `QdrantVectorStore`（在线）
  - `FaissSqliteVectorStore`（离线）
- 兼容服务层：
  - `HybridRetriever`（dense+sparse 融合）
  - `VectorInfrastructureService`（统一门面）

## 4. 在线与离线差异处理

- 在线（Qdrant）
  - 三分区按 collection 隔离：`knowledge_cyber/ai/crypto`
  - payload 保存文本与元数据
  - payload index 覆盖常见过滤字段
- 离线（FAISS + SQLite）
  - SQLite 管理 chunk 原文、元数据、删除标记、FTS5
  - FAISS 管理向量 ANN（Phase 1 使用 `IndexFlatIP`）
  - 删除策略：软删 + 重建索引

## 5. Hybrid 策略

`hybrid_search` 固定在兼容层实现，避免深绑任一后端：

1. dense: `VectorStore.search(...)`
2. sparse: BM25（在线内存 BM25 / 离线 SQLite FTS5）
3. fusion: RRF（Reciprocal Rank Fusion）

## 6. 与 Vue 前端的接入方式

前端不直接调用 `stores/*`。建议链路保持：

`Vue(4.6) -> FastAPI(4.5) -> VectorInfrastructureService(4.3)`

后续 4.5 只需要把 HTTP 入参映射到 `VectorInfrastructureService` 方法即可，不需要感知底层是 Qdrant 还是 FAISS。
