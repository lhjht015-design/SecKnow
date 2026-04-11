from __future__ import annotations

import re
import sqlite3
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from secknow.vector_store.config import assert_zone
from secknow.vector_store.models import (
    ChunkRecord,
    SparseHit,
    ZoneId,
    generate_chunk_id,
)
from secknow.vector_store.stores.base import normalize_search_filters

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - 依赖运行环境
    BM25Okapi = None


class SparseTextIndex(ABC):
    @abstractmethod
    def upsert(self, zone_id: ZoneId, records: list[ChunkRecord]) -> None: ...

    @abstractmethod
    def delete(self, zone_id: ZoneId, chunk_ids: list[str]) -> None: ...

    @abstractmethod
    def delete_by_doc_id(self, zone_id: ZoneId, doc_id: str) -> None: ...

    @abstractmethod
    def search(
        self,
        zone_id: ZoneId,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SparseHit]: ...


class MemoryBm25Index(SparseTextIndex):
    """使用 rank-bm25 的简易在线稀疏索引。"""

    def __init__(self) -> None:
        if BM25Okapi is None:
            raise RuntimeError(
                "rank-bm25 is required for MemoryBm25Index. Install with: pip install rank-bm25"
            )
        self._docs: dict[str, dict[str, tuple[str, str, dict[str, Any]]]] = defaultdict(
            dict
        )
        self._models: dict[
            str, tuple[Any, list[str], list[str], list[str], list[dict[str, Any]]]
        ] = {}

    def upsert(self, zone_id: ZoneId, records: list[ChunkRecord]) -> None:
        assert_zone(zone_id)
        if not records:
            return
        zone_docs = self._docs[zone_id]
        for record in records:
            metadata = record.metadata
            chunk_id = metadata.chunk_id or generate_chunk_id(metadata)
            metadata.chunk_id = chunk_id
            zone_docs[chunk_id] = (
                record.text,
                metadata.doc_id,
                metadata.model_dump(),
            )
        self._rebuild(zone_id)

    def delete(self, zone_id: ZoneId, chunk_ids: list[str]) -> None:
        assert_zone(zone_id)
        if not chunk_ids:
            return
        zone_docs = self._docs[zone_id]
        for chunk_id in chunk_ids:
            zone_docs.pop(chunk_id, None)
        self._rebuild(zone_id)

    def delete_by_doc_id(self, zone_id: ZoneId, doc_id: str) -> None:
        assert_zone(zone_id)
        zone_docs = self._docs[zone_id]
        to_delete = [
            chunk_id
            for chunk_id, (_, _, metadata) in zone_docs.items()
            if metadata.get("doc_id") == doc_id
        ]
        for chunk_id in to_delete:
            zone_docs.pop(chunk_id, None)
        self._rebuild(zone_id)

    def search(
        self,
        zone_id: ZoneId,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SparseHit]:
        assert_zone(zone_id)
        if top_k <= 0 or not query.strip():
            return []
        model = self._models.get(zone_id)
        if model is None:
            return []

        normalized_filters = normalize_search_filters(filters)
        bm25, chunk_ids, texts, doc_ids, metadatas = model
        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = np.asarray(bm25.get_scores(tokens), dtype=np.float32)
        if scores.size == 0:
            return []

        top_indices = np.argsort(-scores)
        hits: list[SparseHit] = []
        for idx in top_indices.tolist():
            if not _metadata_matches(metadatas[idx], normalized_filters):
                continue
            hits.append(
                SparseHit(
                    chunk_id=chunk_ids[idx],
                    doc_id=doc_ids[idx],
                    zone_id=zone_id,
                    score=float(scores[idx]),
                    text=texts[idx],
                    metadata=_public_metadata(metadatas[idx]),
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def _rebuild(self, zone_id: ZoneId) -> None:
        zone_docs = self._docs[zone_id]
        if not zone_docs:
            self._models.pop(zone_id, None)
            return

        chunk_ids = list(zone_docs.keys())
        texts = [zone_docs[c][0] for c in chunk_ids]
        doc_ids = [zone_docs[c][1] for c in chunk_ids]
        metadatas = [zone_docs[c][2] for c in chunk_ids]
        tokenized = [_tokenize(text) for text in texts]
        self._models[zone_id] = (
            BM25Okapi(tokenized),
            chunk_ids,
            texts,
            doc_ids,
            metadatas,
        )


class SQLiteFtsSparseIndex(SparseTextIndex):
    """委托给 SQLite FTS5 + bm25() 排序的稀疏索引。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()

    def upsert(self, zone_id: ZoneId, records: list[ChunkRecord]) -> None:
        # FTS 由 chunks 表上的 SQLite 触发器自动维护。
        assert_zone(zone_id)
        _ = records

    def delete(self, zone_id: ZoneId, chunk_ids: list[str]) -> None:
        # FTS 同步由更新触发器处理。
        assert_zone(zone_id)
        _ = chunk_ids

    def delete_by_doc_id(self, zone_id: ZoneId, doc_id: str) -> None:
        # FTS 同步由 chunks 表变更触发器处理。
        assert_zone(zone_id)
        _ = doc_id

    def search(
        self,
        zone_id: ZoneId,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SparseHit]:
        assert_zone(zone_id)
        if top_k <= 0 or not query.strip():
            return []

        normalized_filters = normalize_search_filters(filters)
        params: list[Any] = [_sqlite_fts_query(query), zone_id]
        sql = """
            SELECT c.chunk_id, c.doc_id, c.zone_id, c.text, c.filename, c.source_path, c.extension,
                   c.chunk_index, c.chunk_count, c.char_len, c.content_hash, c.mtime, c.size_bytes,
                   c.file_type, c.language, c.record_type, bm25(chunk_fts) AS bm25_score
            FROM chunk_fts
            JOIN chunks c ON c.id = chunk_fts.rowid
            WHERE chunk_fts MATCH ? AND c.zone_id = ? AND c.is_deleted = 0
        """
        for key, value in normalized_filters.items():
            sql += f" AND c.{key} = ?"
            params.append(value)
        sql += " ORDER BY bm25_score LIMIT ?"
        params.append(top_k)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()

        hits: list[SparseHit] = []
        for row in rows:
            hits.append(
                SparseHit(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    zone_id=zone_id,
                    score=float(-row["bm25_score"]),
                    text=row["text"],
                    metadata=_public_metadata(dict(row)),
                )
            )
        return hits


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _sqlite_fts_query(text: str) -> str:
    # 第一阶段刻意保持查询解析器保守，先保证稳定性。
    terms = [t for t in _tokenize(text) if t]
    return " OR ".join(terms) if terms else text.strip()


def _metadata_matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(metadata.get(key) == value for key, value in filters.items())


def _public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "filename": metadata.get("filename"),
        "source_path": metadata.get("source_path"),
        "extension": metadata.get("extension"),
        "chunk_index": metadata.get("chunk_index"),
        "chunk_count": metadata.get("chunk_count"),
        "char_len": metadata.get("char_len"),
        "content_hash": metadata.get("content_hash"),
        "mtime": metadata.get("mtime"),
        "size_bytes": metadata.get("size_bytes"),
        "file_type": metadata.get("file_type"),
        "language": metadata.get("language"),
        "record_type": metadata.get("record_type"),
    }
