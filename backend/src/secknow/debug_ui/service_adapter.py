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
    doc_count: int
    filename_count: int
    filenames: list[str]
    record_type_counts: dict[str, int]
    filename_chunk_counts: list[tuple[str, int]]


@dataclass(slots=True)
class FileChunkDebug:
    """单个文件 chunk 的完整调试信息。"""

    collection_name: str
    chunk_id: str
    doc_id: str
    filename: str
    source_path: str
    extension: str
    file_type: str
    language: str | None
    record_type: str
    chunk_index: int
    chunk_count: int
    char_len: int
    content_hash: str
    mtime: int
    size_bytes: int
    text: str
    vector_dim: int
    vector_preview: list[float]


@dataclass(slots=True)
class FileMetadataDebugResult:
    """按文件名或 doc_id 查询到的文件元信息结果。"""

    query_doc_id: str | None
    query_filename: str | None
    chunk_total: int
    matched_doc_ids: list[str]
    matched_filenames: list[str]
    chunks: list[FileChunkDebug]


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
    """执行仅依赖 Qdrant 稠密检索的语义检索调试流程。"""
    vector_service = build_service(settings)
    encoder = QueryEncoder(
        embedding_mode=settings.embedding_mode,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )
    query_vector = encoder.encode(query)
    raw_hits = vector_service.search(
        zone_id=zone_id,
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
        filenames: list[str] = []
        record_type_counts: dict[str, int] = {}
        doc_ids: set[str] = set()
        filename_chunk_counter: dict[str, int] = {}
        points, _ = client.scroll(
            collection_name=name,
            with_payload=True,
            with_vectors=False,
            limit=1,
        )
        if points:
            sample_payload = dict(points[0].payload or {})
            sample_payload_keys = sorted(sample_payload.keys())

        offset = None
        filename_set: set[str] = set()
        while True:
            batch, offset = client.scroll(
                collection_name=name,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for point in batch:
                payload = dict(point.payload or {})
                doc_id = payload.get("doc_id")
                filename = payload.get("filename")
                record_type = payload.get("record_type", "unknown")
                if doc_id:
                    doc_ids.add(str(doc_id))
                if filename:
                    filename_set.add(str(filename))
                    filename_chunk_counter[str(filename)] = (
                        filename_chunk_counter.get(str(filename), 0) + 1
                    )
                record_type_counts[str(record_type)] = (
                    record_type_counts.get(str(record_type), 0) + 1
                )
            if offset is None:
                break
        filenames = sorted(filename_set)

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
                doc_count=len(doc_ids),
                filename_count=len(filenames),
                filenames=filenames,
                record_type_counts=record_type_counts,
                filename_chunk_counts=sorted(
                    filename_chunk_counter.items(),
                    key=lambda item: (-item[1], item[0]),
                ),
            )
        )

    return QdrantDebugResult(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collections=items,
    )


def inspect_file_metadata(
    settings: UiSettings,
    *,
    doc_id: str | None = None,
    filename: str | None = None,
) -> FileMetadataDebugResult:
    """按 doc_id 或 filename 读取某个文件的全部入库信息。"""
    if settings.mode != "online":
        raise ValueError("文件元信息页当前只支持 online 模式下的 Qdrant。")

    query_doc_id = (doc_id or "").strip() or None
    query_filename = (filename or "").strip() or None
    if query_doc_id is None and query_filename is None:
        raise ValueError("请至少提供 doc_id 或 filename。")

    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    collections_resp = client.get_collections()

    conditions = []
    if query_doc_id is not None:
        conditions.append(
            qm.FieldCondition(
                key="doc_id",
                match=qm.MatchValue(value=query_doc_id),
            )
        )
    if query_filename is not None:
        conditions.append(
            qm.FieldCondition(
                key="filename",
                match=qm.MatchValue(value=query_filename),
            )
        )
    query_filter = qm.Filter(must=conditions)

    chunks: list[FileChunkDebug] = []
    matched_doc_ids: set[str] = set()
    matched_filenames: set[str] = set()

    for collection in collections_resp.collections:
        offset = None
        while True:
            batch, offset = client.scroll(
                collection_name=collection.name,
                scroll_filter=query_filter,
                with_payload=True,
                with_vectors=True,
                limit=256,
                offset=offset,
            )
            for point in batch:
                payload = dict(point.payload or {})
                vector = point.vector
                if isinstance(vector, dict):
                    vector = next(iter(vector.values()), [])
                vector_list = list(vector or [])

                matched_doc_ids.add(str(payload.get("doc_id", "")))
                matched_filenames.add(str(payload.get("filename", "")))
                chunks.append(
                    FileChunkDebug(
                        collection_name=collection.name,
                        chunk_id=str(payload.get("chunk_id", point.id)),
                        doc_id=str(payload.get("doc_id", "")),
                        filename=str(payload.get("filename", "")),
                        source_path=str(payload.get("source_path", "")),
                        extension=str(payload.get("extension", "")),
                        file_type=str(payload.get("file_type", "")),
                        language=payload.get("language"),
                        record_type=str(payload.get("record_type", "")),
                        chunk_index=int(payload.get("chunk_index", 0)),
                        chunk_count=int(payload.get("chunk_count", 0)),
                        char_len=int(payload.get("char_len", 0)),
                        content_hash=str(payload.get("content_hash", "")),
                        mtime=int(payload.get("mtime", 0)),
                        size_bytes=int(payload.get("size_bytes", 0)),
                        text=str(payload.get("text", "")),
                        vector_dim=len(vector_list),
                        vector_preview=vector_list[:8],
                    )
                )
            if offset is None:
                break

    chunks.sort(
        key=lambda item: (
            item.collection_name,
            item.doc_id,
            item.chunk_index,
            item.chunk_id,
        )
    )
    return FileMetadataDebugResult(
        query_doc_id=query_doc_id,
        query_filename=query_filename,
        chunk_total=len(chunks),
        matched_doc_ids=sorted(item for item in matched_doc_ids if item),
        matched_filenames=sorted(item for item in matched_filenames if item),
        chunks=chunks,
    )
