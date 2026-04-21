from __future__ import annotations

"""调试 UI 使用的服务适配层。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from secknow.inference.services.formatter import SearchResultFormatter
from secknow.inference.services.query_encoder import QueryEncoder
from secknow.inference.services.reranker import SearchReranker
from secknow.text_processing import DocumentTextPipeline
from secknow.text_processing.schemas.pipeline_result import PipelineResult
from secknow.vector_store.factory import build_vector_service
from secknow.vector_store.models import SearchHit, UpsertResult


@dataclass(slots=True)
class UiSettings:
    """调试 UI 的连接与模型配置。"""

    mode: str
    qdrant_host: str
    qdrant_port: int
    sparse_db_path: str
    db_path: str
    index_dir: str
    embedding_mode: str
    embedding_model: str
    embedding_dim: int


@dataclass(slots=True)
class IngestDebugResult:
    """入库调试结果。"""

    pipeline_result: PipelineResult
    upsert_result: UpsertResult | None


@dataclass(slots=True)
class SearchDebugResult:
    """语义检索调试结果。"""

    query: str
    query_vector: list[float]
    raw_hits: list[SearchHit]
    formatted_results: list[Any]


@dataclass(slots=True)
class QdrantCollectionDebug:
    """单个 Qdrant collection 的结构化调试信息。"""

    name: str
    points_count: int | None
    vectors_count: int | None
    indexed_vectors_count: int | None
    vector_size: int | None
    distance: str | None
    payload_schema: dict[str, Any]
    sample_payload_keys: list[str]
    sample_payload: dict[str, Any]


@dataclass(slots=True)
class QdrantDebugResult:
    """Qdrant 结构调试结果。"""

    host: str
    port: int
    collections: list[QdrantCollectionDebug]


def build_service(settings: UiSettings):
    """按 UI 配置构建 4.3 服务。"""
    if settings.mode == "online":
        return build_vector_service(
            mode="online",
            qdrant_host=settings.qdrant_host,
            qdrant_port=settings.qdrant_port,
            embedding_model=settings.embedding_model,
            sparse_db_path=settings.sparse_db_path,
        )
    return build_vector_service(
        mode="offline",
        db_path=settings.db_path,
        index_dir=settings.index_dir,
        embedding_dim=settings.embedding_dim,
        embedding_model=settings.embedding_model,
    )


def ingest_file(
    *,
    settings: UiSettings,
    file_path: str,
    zone_id: str,
    record_type: str,
    chunk_strategy: str,
    dedup_strategy: str,
    max_tokens: int,
    overlap: int,
    language: str | None,
    write_to_store: bool,
) -> IngestDebugResult:
    """执行单文件入库调试流程。"""
    pipeline = DocumentTextPipeline(
        chunk_strategy=chunk_strategy,
        dedup_strategy=dedup_strategy,
        max_tokens=max_tokens,
        overlap=overlap,
        embedding_mode=settings.embedding_mode,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )
    result = pipeline.process_file(
        file_path=Path(file_path),
        zone_id=zone_id,
        record_type=record_type,
        language=language or None,
        return_result=True,
    )
    assert isinstance(result, PipelineResult)

    if not write_to_store or not result.records:
        return IngestDebugResult(pipeline_result=result, upsert_result=None)

    vector_service = build_service(settings)
    upsert_result = vector_service.upsert(zone_id=zone_id, records=result.records)
    return IngestDebugResult(pipeline_result=result, upsert_result=upsert_result)


def semantic_search(
    *,
    settings: UiSettings,
    query: str,
    zone_id: str,
    top_k: int,
    record_type: str,
) -> SearchDebugResult:
    """执行语义检索调试流程。"""
    vector_service = build_service(settings)
    encoder = QueryEncoder(
        embedding_mode=settings.embedding_mode,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )
    query_vector = encoder.encode(query)
    raw_hits = vector_service.hybrid_search(
        zone_id=zone_id,
        query=query,
        query_vec=query_vector,
        top_k=top_k,
        filters={"record_type": record_type},
    )
    reranked = SearchReranker().rerank(query=query, hits=raw_hits)
    formatted = SearchResultFormatter().format(reranked)

    return SearchDebugResult(
        query=query,
        query_vector=query_vector,
        raw_hits=raw_hits,
        formatted_results=formatted,
    )


def inspect_qdrant(settings: UiSettings) -> QdrantDebugResult:
    """读取 Qdrant 的 collection 结构信息。"""
    if settings.mode != "online":
        raise ValueError("数据库结构页当前只支持 online 模式下的 Qdrant。")

    from qdrant_client import QdrantClient

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    collections_resp = client.get_collections()
    items: list[QdrantCollectionDebug] = []

    for collection in collections_resp.collections:
        name = collection.name
        info = client.get_collection(collection_name=name)

        vectors_config = getattr(getattr(info, "config", None), "params", None)
        vector_size = getattr(getattr(vectors_config, "vectors", None), "size", None)
        distance = getattr(getattr(vectors_config, "vectors", None), "distance", None)

        sample_payload: dict[str, Any] = {}
        sample_payload_keys: list[str] = []
        points, _ = client.scroll(
            collection_name=name,
            with_payload=True,
            with_vectors=False,
            limit=1,
        )
        if points:
            sample_payload = dict(points[0].payload or {})
            sample_payload_keys = sorted(sample_payload.keys())

        payload_schema_raw = getattr(info, "payload_schema", None) or {}
        payload_schema = {
            key: str(value)
            for key, value in dict(payload_schema_raw).items()
        }

        items.append(
            QdrantCollectionDebug(
                name=name,
                points_count=getattr(info, "points_count", None),
                vectors_count=getattr(info, "vectors_count", None),
                indexed_vectors_count=getattr(info, "indexed_vectors_count", None),
                vector_size=vector_size,
                distance=str(distance) if distance is not None else None,
                payload_schema=payload_schema,
                sample_payload_keys=sample_payload_keys,
                sample_payload=sample_payload,
            )
        )

    return QdrantDebugResult(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collections=items,
    )
