# 4.3 Vector Store（Phase 2 进度）

本目录实现 4.3 向量存储层的在线/离线统一能力，并在 Phase 2 完成统一过滤契约、文档级删除与导出统计口径对齐。

## 当前状态

Phase 1（基础设施）已完成：

- 统一数据模型：`ChunkRecord / SearchHit / SparseHit / BaselineBundle / ExportManifest`
- 统一门面：`VectorInfrastructureService`
- 在线后端：`QdrantVectorStore`
- 离线后端：`FaissSqliteVectorStore` + `SQLiteFtsSparseIndex`
- 融合检索：`HybridRetriever`（dense + sparse, RRF 融合）

Phase 2（契约统一）已完成：

- 新增文档级删除接口：`delete_by_doc_id(zone_id, doc_id)`
- 统一检索过滤字段白名单：
  - `doc_id, filename, source_path, extension, file_type, language, record_type`
- 非法过滤字段统一忽略，不抛错
- `search()` / `hybrid_search()` 默认限定 `record_type=knowledge`
- 仅当显式 `filters={"record_type":"baseline"}` 时允许普通检索返回 baseline
- `get_baseline()` 始终只返回 baseline
- `export_zone()` 统一统计口径：
  - `record_count` = knowledge 数量
  - `baseline_count` = baseline 数量
  - 在线/离线返回字典均显式包含这两个字段
- 修复在线 `delete_by_doc_id` 行为，确保同 `doc_id` 下 knowledge 与 baseline 都能删除

## 对外接口（4.5/4.4 调用）

统一通过 `VectorInfrastructureService` 暴露：

- `ensure_zone(zone_id, dim)`
- `upsert(zone_id, records)`
- `search(zone_id, query_vec, top_k=10, filters=None)`
- `hybrid_search(zone_id, query, query_vec, top_k=10, filters=None)`
- `delete(zone_id, chunk_ids)`
- `delete_by_doc_id(zone_id, doc_id)`
- `export_zone(zone_id, target_dir)`
- `get_baseline(zone_id)`

## 模块结构

- 模型与契约：`src/secknow/vector_store/models.py`
- 抽象与过滤契约：`src/secknow/vector_store/stores/base.py`
- 在线实现：`src/secknow/vector_store/stores/qdrant_store.py`
- 离线实现：`src/secknow/vector_store/stores/faiss_sqlite_store.py`
- 稀疏检索：`src/secknow/vector_store/services/sparse.py`
- 融合层：`src/secknow/vector_store/services/hybrid.py`
- 统一门面：`src/secknow/vector_store/services/vector_service.py`

## 测试进度（Phase 2）

新增契约测试文件：

- `src/secknow/vector_store/tests/test_unified_contract.py`

覆盖点：

- offline dense/sparse/hybrid 默认仅返回 knowledge
- 显式 baseline 过滤时 dense/sparse/hybrid 可查 baseline
- 非法过滤字段被忽略
- offline / online `delete_by_doc_id`
- offline / online `export_zone` 统计字段
- `get_baseline` 仅返回 baseline

## 调用链路

建议保持：

`Vue(4.6) -> FastAPI(4.5) -> VectorInfrastructureService(4.3)`

4.5 只需映射 HTTP 入参与返回值，不应直接耦合 `stores/*` 实现细节。
