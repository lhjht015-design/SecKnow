from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from secknow.vector_store.models import ChunkMetadata, ChunkRecord
from secknow.vector_store.services.sparse import MemoryBm25Index, SQLiteFtsSparseIndex
from secknow.vector_store.services.vector_service import VectorInfrastructureService
from secknow.vector_store.stores.faiss_sqlite_store import FaissSqliteVectorStore
from secknow.vector_store.stores.qdrant_store import QdrantVectorStore


ZONE_ID = "cyber"
EMBED_DIM = 4


def _record(
    *,
    doc_id: str,
    text: str,
    vector: list[float],
    chunk_index: int = 0,
    record_type: str = "knowledge",
) -> ChunkRecord:
    return ChunkRecord(
        text=text,
        vector=vector,
        metadata=ChunkMetadata(
            doc_id=doc_id,
            zone_id=ZONE_ID,
            filename=f"{doc_id}.md",
            source_path=f"/tmp/{doc_id}.md",
            extension=".md",
            chunk_index=chunk_index,
            chunk_count=1,
            char_len=len(text),
            content_hash=f"h-{doc_id}-{chunk_index}-{record_type}",
            mtime=1700000000,
            size_bytes=len(text.encode("utf-8")),
            file_type="markdown",
            language="zh",
            record_type=record_type,
        ),
    )


def _mixed_records() -> list[ChunkRecord]:
    return [
        _record(
            doc_id="doc-knowledge",
            text="network defense knowledge alpha",
            vector=[0.8, 0.2, 0.0, 0.0],
            record_type="knowledge",
        ),
        _record(
            doc_id="doc-baseline",
            text="network defense baseline controls",
            vector=[1.0, 0.0, 0.0, 0.0],
            record_type="baseline",
        ),
    ]


def _build_offline_service(tmp_path: Path) -> VectorInfrastructureService:
    store = FaissSqliteVectorStore(
        db_path=tmp_path / "secknow.sqlite3",
        index_dir=tmp_path / "faiss",
        embedding_dim=EMBED_DIM,
    )
    return VectorInfrastructureService(
        dense_store=store,
        sparse_index=SQLiteFtsSparseIndex(tmp_path / "secknow.sqlite3"),
    )


@pytest.fixture
def online_service() -> VectorInfrastructureService:
    prefix = f"test43-{uuid4().hex[:8]}"
    dense_store = QdrantVectorStore(
        host="localhost",
        port=6333,
        collection_prefix=prefix,
    )
    service = VectorInfrastructureService(dense_store=dense_store, sparse_index=MemoryBm25Index())
    collection_name = f"{prefix}_{ZONE_ID}"

    try:
        service.ensure_zone(zone_id=ZONE_ID, dim=EMBED_DIM)
    except Exception as exc:  # pragma: no cover - 依赖本地环境
        pytest.skip(f"Qdrant unavailable: {exc}")

    try:
        yield service
    finally:
        try:
            dense_store.client.delete_collection(collection_name=collection_name)
        except Exception:
            pass


def test_offline_default_searches_only_knowledge(tmp_path: Path) -> None:
    service = _build_offline_service(tmp_path)
    service.upsert(zone_id=ZONE_ID, records=_mixed_records())

    dense_hits = service.search(zone_id=ZONE_ID, query_vec=[1.0, 0.0, 0.0, 0.0], top_k=5)
    sparse_hits = service.sparse_index.search(zone_id=ZONE_ID, query="network defense", top_k=5)
    hybrid_hits = service.hybrid_search(
        zone_id=ZONE_ID,
        query="network defense",
        query_vec=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
    )

    assert dense_hits and all(h.metadata.get("record_type") == "knowledge" for h in dense_hits)
    assert sparse_hits and all(h.metadata.get("record_type") == "knowledge" for h in sparse_hits)
    assert hybrid_hits and all(h.metadata.get("record_type") == "knowledge" for h in hybrid_hits)


def test_offline_explicit_baseline_filter_works_for_dense_sparse_hybrid(tmp_path: Path) -> None:
    service = _build_offline_service(tmp_path)
    service.upsert(zone_id=ZONE_ID, records=_mixed_records())

    dense_hits = service.search(
        zone_id=ZONE_ID,
        query_vec=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
        filters={"record_type": "baseline"},
    )
    sparse_hits = service.sparse_index.search(
        zone_id=ZONE_ID,
        query="network defense",
        top_k=5,
        filters={"record_type": "baseline"},
    )
    hybrid_hits = service.hybrid_search(
        zone_id=ZONE_ID,
        query="network defense",
        query_vec=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
        filters={"record_type": "baseline"},
    )

    assert dense_hits and all(h.metadata.get("record_type") == "baseline" for h in dense_hits)
    assert sparse_hits and all(h.metadata.get("record_type") == "baseline" for h in sparse_hits)
    assert hybrid_hits and all(h.metadata.get("record_type") == "baseline" for h in hybrid_hits)


def test_offline_illegal_filter_fields_are_ignored(tmp_path: Path) -> None:
    service = _build_offline_service(tmp_path)
    service.upsert(zone_id=ZONE_ID, records=_mixed_records())

    dense_expected = service.search(
        zone_id=ZONE_ID,
        query_vec=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
        filters={"record_type": "knowledge"},
    )
    dense_actual = service.search(
        zone_id=ZONE_ID,
        query_vec=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
        filters={"record_type": "knowledge", "bad_field": "ignored"},
    )
    sparse_expected = service.sparse_index.search(
        zone_id=ZONE_ID,
        query="network defense",
        top_k=5,
        filters={"record_type": "knowledge"},
    )
    sparse_actual = service.sparse_index.search(
        zone_id=ZONE_ID,
        query="network defense",
        top_k=5,
        filters={"record_type": "knowledge", "bad_field": "ignored"},
    )
    hybrid_expected = service.hybrid_search(
        zone_id=ZONE_ID,
        query="network defense",
        query_vec=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
        filters={"record_type": "knowledge"},
    )
    hybrid_actual = service.hybrid_search(
        zone_id=ZONE_ID,
        query="network defense",
        query_vec=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
        filters={"record_type": "knowledge", "bad_field": "ignored"},
    )

    assert [h.chunk_id for h in dense_actual] == [h.chunk_id for h in dense_expected]
    assert [h.chunk_id for h in sparse_actual] == [h.chunk_id for h in sparse_expected]
    assert [h.chunk_id for h in hybrid_actual] == [h.chunk_id for h in hybrid_expected]


def test_offline_delete_by_doc_id(tmp_path: Path) -> None:
    service = _build_offline_service(tmp_path)
    service.upsert(
        zone_id=ZONE_ID,
        records=[
            _record(doc_id="doc-delete", text="to delete one", vector=[1.0, 0.0, 0.0, 0.0]),
            _record(
                doc_id="doc-delete",
                text="to delete two",
                vector=[0.9, 0.1, 0.0, 0.0],
                chunk_index=1,
            ),
            _record(doc_id="doc-keep", text="keep me", vector=[0.0, 1.0, 0.0, 0.0]),
        ],
    )
    result = service.delete_by_doc_id(zone_id=ZONE_ID, doc_id="doc-delete")

    deleted_hits = service.search(
        zone_id=ZONE_ID,
        query_vec=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
        filters={"doc_id": "doc-delete", "record_type": "knowledge"},
    )
    keep_hits = service.search(
        zone_id=ZONE_ID,
        query_vec=[0.0, 1.0, 0.0, 0.0],
        top_k=5,
        filters={"doc_id": "doc-keep"},
    )

    assert result.requested == 2
    assert result.deleted == 2
    assert deleted_hits == []
    assert keep_hits and all(h.doc_id == "doc-keep" for h in keep_hits)


def test_online_delete_by_doc_id(online_service: VectorInfrastructureService) -> None:
    online_service.upsert(
        zone_id=ZONE_ID,
        records=[
            _record(doc_id="doc-delete-online", text="online delete one", vector=[1.0, 0.0, 0.0, 0.0]),
            _record(
                doc_id="doc-delete-online",
                text="online delete two",
                vector=[0.95, 0.05, 0.0, 0.0],
                chunk_index=1,
                record_type="baseline",
            ),
            _record(doc_id="doc-keep-online", text="online keep", vector=[0.0, 1.0, 0.0, 0.0]),
        ],
    )
    result = online_service.delete_by_doc_id(zone_id=ZONE_ID, doc_id="doc-delete-online")

    knowledge_hits = online_service.search(
        zone_id=ZONE_ID,
        query_vec=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
        filters={"doc_id": "doc-delete-online", "record_type": "knowledge"},
    )
    baseline_hits = online_service.search(
        zone_id=ZONE_ID,
        query_vec=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
        filters={"doc_id": "doc-delete-online", "record_type": "baseline"},
    )

    assert result.requested == 2
    assert result.deleted == 2
    assert knowledge_hits == []
    assert baseline_hits == []


def test_offline_export_zone_counts(tmp_path: Path) -> None:
    service = _build_offline_service(tmp_path)
    service.upsert(
        zone_id=ZONE_ID,
        records=[
            _record(doc_id="doc-k1", text="k1", vector=[1.0, 0.0, 0.0, 0.0], record_type="knowledge"),
            _record(doc_id="doc-k2", text="k2", vector=[0.0, 1.0, 0.0, 0.0], record_type="knowledge"),
            _record(doc_id="doc-b1", text="b1", vector=[1.0, 0.0, 0.0, 0.0], record_type="baseline"),
        ],
    )

    result = service.export_zone(zone_id=ZONE_ID, target_dir=str(tmp_path / "export"))
    manifest = json.loads(Path(result["manifest_file"]).read_text(encoding="utf-8"))

    assert "record_count" in result
    assert "baseline_count" in result
    assert result["record_count"] == 2
    assert result["baseline_count"] == 1
    assert manifest["record_count"] == 2
    assert manifest["baseline_count"] == 1


def test_online_export_zone_counts(online_service: VectorInfrastructureService, tmp_path: Path) -> None:
    online_service.upsert(
        zone_id=ZONE_ID,
        records=[
            _record(doc_id="doc-k1-online", text="k1", vector=[1.0, 0.0, 0.0, 0.0], record_type="knowledge"),
            _record(doc_id="doc-k2-online", text="k2", vector=[0.0, 1.0, 0.0, 0.0], record_type="knowledge"),
            _record(doc_id="doc-b1-online", text="b1", vector=[1.0, 0.0, 0.0, 0.0], record_type="baseline"),
        ],
    )
    result = online_service.export_zone(zone_id=ZONE_ID, target_dir=str(tmp_path / "export"))
    manifest = json.loads(Path(result["manifest_file"]).read_text(encoding="utf-8"))

    assert "record_count" in result
    assert "baseline_count" in result
    assert result["record_count"] == 2
    assert result["baseline_count"] == 1
    assert manifest["record_count"] == 2
    assert manifest["baseline_count"] == 1


def test_get_baseline_returns_only_baseline(tmp_path: Path) -> None:
    service = _build_offline_service(tmp_path)
    service.upsert(zone_id=ZONE_ID, records=_mixed_records())

    baseline = service.get_baseline(zone_id=ZONE_ID)
    assert baseline.count == 1
    assert all(item.get("record_type") == "baseline" for item in baseline.metadatas)
