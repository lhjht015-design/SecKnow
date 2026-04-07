from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

from secknow.vector_store.config import assert_zone, utc_now_iso
from secknow.vector_store.models import (
    BaselineBundle,
    ChunkRecord,
    DeleteResult,
    ExportManifest,
    SearchHit,
    UpsertResult,
    ZoneId,
    generate_chunk_id,
)
from secknow.vector_store.stores.base import VectorStore

try:
    import faiss
except ImportError:  # pragma: no cover - 依赖运行环境
    faiss = None


class FaissSqliteVectorStore(VectorStore):
    """离线向量存储后端（SQLite 元数据 + FAISS ANN 索引）。"""

    _FILTERABLE_FIELDS = {
        "doc_id",
        "filename",
        "source_path",
        "extension",
        "file_type",
        "language",
        "record_type",
    }

    def __init__(
        self,
        db_path: str | Path = "db/secknow.sqlite3",
        index_dir: str | Path = "db/faiss",
        embedding_dim: int = 384,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        distance: str = "cosine",
    ) -> None:
        if faiss is None:
            raise RuntimeError(
                "faiss is required for FaissSqliteVectorStore. Install with: pip install faiss-cpu"
            )

        self.db_path = Path(db_path).expanduser().resolve()
        self.index_dir = Path(index_dir).expanduser().resolve()
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self.distance = distance.lower()
        self.chunk_strategy = {"type": "hybrid", "max_tokens": 500, "overlap": 50}

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._indexes: dict[str, Any] = {}
        self._init_schema()

    def ensure_zone(self, zone_id: ZoneId, dim: int) -> None:
        assert_zone(zone_id)
        if dim != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.embedding_dim}, got {dim}"
            )
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO zones (zone_id, embedding_dim, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(zone_id) DO UPDATE SET embedding_dim = excluded.embedding_dim
                """,
                (zone_id, dim, now),
            )
            conn.commit()

    def upsert(self, zone_id: ZoneId, records: list[ChunkRecord]) -> UpsertResult:
        assert_zone(zone_id)
        if not records:
            return UpsertResult(zone_id=zone_id, attempted=0, inserted=0, chunk_ids=[])

        self.ensure_zone(zone_id, len(records[0].vector))
        now = int(time.time())
        chunk_ids: list[str] = []

        with self._connect() as conn:
            for record in records:
                metadata = record.metadata
                if metadata.zone_id != zone_id:
                    raise ValueError(
                        f"Record zone mismatch: metadata.zone_id={metadata.zone_id}, upsert zone={zone_id}"
                    )
                if len(record.vector) != self.embedding_dim:
                    raise ValueError(
                        f"Record vector dim mismatch: expected {self.embedding_dim}, got {len(record.vector)}"
                    )

                chunk_id = metadata.chunk_id or generate_chunk_id(metadata)
                metadata.chunk_id = chunk_id
                chunk_ids.append(chunk_id)

                conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, doc_id, zone_id, record_type, text, filename, source_path, extension,
                        chunk_index, chunk_count, char_len, content_hash, mtime, size_bytes, file_type,
                        language, is_deleted, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        doc_id = excluded.doc_id,
                        zone_id = excluded.zone_id,
                        record_type = excluded.record_type,
                        text = excluded.text,
                        filename = excluded.filename,
                        source_path = excluded.source_path,
                        extension = excluded.extension,
                        chunk_index = excluded.chunk_index,
                        chunk_count = excluded.chunk_count,
                        char_len = excluded.char_len,
                        content_hash = excluded.content_hash,
                        mtime = excluded.mtime,
                        size_bytes = excluded.size_bytes,
                        file_type = excluded.file_type,
                        language = excluded.language,
                        is_deleted = 0,
                        updated_at = excluded.updated_at
                    """,
                    (
                        chunk_id,
                        metadata.doc_id,
                        metadata.zone_id,
                        metadata.record_type,
                        record.text,
                        metadata.filename,
                        metadata.source_path,
                        metadata.extension,
                        metadata.chunk_index,
                        metadata.chunk_count,
                        metadata.char_len,
                        metadata.content_hash,
                        metadata.mtime,
                        metadata.size_bytes,
                        metadata.file_type,
                        metadata.language,
                        now,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO chunk_vectors (chunk_id, zone_id, vector, dim, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        zone_id = excluded.zone_id,
                        vector = excluded.vector,
                        dim = excluded.dim,
                        updated_at = excluded.updated_at
                    """,
                    (
                        chunk_id,
                        metadata.zone_id,
                        self._vector_to_blob(record.vector),
                        self.embedding_dim,
                        now,
                    ),
                )
            conn.commit()

        self._rebuild_zone_index(zone_id)
        return UpsertResult(
            zone_id=zone_id,
            attempted=len(records),
            inserted=len(chunk_ids),
            chunk_ids=chunk_ids,
        )

    def search(
        self,
        zone_id: ZoneId,
        query_vec: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        assert_zone(zone_id)
        if len(query_vec) != self.embedding_dim:
            raise ValueError(
                f"Query vector dim mismatch: expected {self.embedding_dim}, got {len(query_vec)}"
            )
        self.ensure_zone(zone_id, self.embedding_dim)
        index = self._load_zone_index(zone_id)
        if index.ntotal == 0:
            return []

        query = self._normalize(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))
        fetch_k = max(top_k * 8, top_k)
        distances, ids = index.search(query, fetch_k)

        candidate_ids = [int(i) for i in ids[0].tolist() if int(i) >= 0]
        if not candidate_ids:
            return []

        rows = self._fetch_rows_by_ids(zone_id, candidate_ids, filters)
        row_map = {int(row["id"]): row for row in rows}

        hits: list[SearchHit] = []
        for idx, score in zip(ids[0].tolist(), distances[0].tolist()):
            row = row_map.get(int(idx))
            if row is None:
                continue
            metadata = {
                "filename": row["filename"],
                "source_path": row["source_path"],
                "extension": row["extension"],
                "chunk_index": row["chunk_index"],
                "chunk_count": row["chunk_count"],
                "char_len": row["char_len"],
                "content_hash": row["content_hash"],
                "mtime": row["mtime"],
                "size_bytes": row["size_bytes"],
                "file_type": row["file_type"],
                "language": row["language"],
                "record_type": row["record_type"],
            }
            hits.append(
                SearchHit(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    zone_id=zone_id,
                    score=float(score),
                    text=row["text"],
                    metadata=metadata,
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def delete(self, zone_id: ZoneId, chunk_ids: list[str]) -> DeleteResult:
        assert_zone(zone_id)
        if not chunk_ids:
            return DeleteResult(zone_id=zone_id, requested=0, deleted=0, chunk_ids=[])

        now = int(time.time())
        placeholders = ",".join(["?"] * len(chunk_ids))
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE chunks
                SET is_deleted = 1, updated_at = ?
                WHERE zone_id = ? AND chunk_id IN ({placeholders})
                """,
                (now, zone_id, *chunk_ids),
            )
            conn.commit()
        self._rebuild_zone_index(zone_id)
        deleted = cursor.rowcount if cursor.rowcount is not None else 0
        return DeleteResult(
            zone_id=zone_id,
            requested=len(chunk_ids),
            deleted=deleted,
            chunk_ids=chunk_ids,
        )

    def get_baseline(self, zone_id: ZoneId) -> BaselineBundle:
        assert_zone(zone_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.chunk_id, c.doc_id, c.filename, c.source_path, c.extension, c.chunk_index,
                       c.chunk_count, c.char_len, c.content_hash, c.mtime, c.size_bytes, c.file_type,
                       c.language, c.record_type, v.vector
                FROM chunks c
                JOIN chunk_vectors v ON c.chunk_id = v.chunk_id
                WHERE c.zone_id = ? AND c.record_type = 'baseline' AND c.is_deleted = 0
                ORDER BY c.id ASC
                """,
                (zone_id,),
            ).fetchall()

        vectors: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        for row in rows:
            vectors.append(self._blob_to_vector(row["vector"]))
            metadatas.append(
                {
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "filename": row["filename"],
                    "source_path": row["source_path"],
                    "extension": row["extension"],
                    "chunk_index": row["chunk_index"],
                    "chunk_count": row["chunk_count"],
                    "char_len": row["char_len"],
                    "content_hash": row["content_hash"],
                    "mtime": row["mtime"],
                    "size_bytes": row["size_bytes"],
                    "file_type": row["file_type"],
                    "language": row["language"],
                    "record_type": row["record_type"],
                }
            )
        return BaselineBundle(zone_id=zone_id, vectors=vectors, metadatas=metadatas)

    def export_zone(self, zone_id: ZoneId, target_dir: str) -> dict[str, Any]:
        assert_zone(zone_id)
        self.ensure_zone(zone_id, self.embedding_dim)
        zone_dir = Path(target_dir).expanduser().resolve() / zone_id
        zone_dir.mkdir(parents=True, exist_ok=True)

        sqlite_out_path = zone_dir / "chunks.sqlite"
        index_out_path = zone_dir / "faiss.index"
        baseline_vec_path = zone_dir / "baseline_vectors.npy"
        baseline_meta_path = zone_dir / "baseline_meta.json"
        manifest_path = zone_dir / "manifest.json"
        checksums_path = zone_dir / "checksums.json"

        if sqlite_out_path.exists():
            sqlite_out_path.unlink()

        self._export_zone_sqlite(zone_id, sqlite_out_path)
        index = self._load_zone_index(zone_id)
        faiss.write_index(index, str(index_out_path))

        baseline = self.get_baseline(zone_id)
        np.save(
            baseline_vec_path,
            np.asarray(baseline.vectors, dtype=np.float32) if baseline.vectors else np.empty((0, self.embedding_dim), dtype=np.float32),
        )
        baseline_meta_path.write_text(
            json.dumps(baseline.metadatas, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        counts = self._zone_counts(zone_id)
        manifest = ExportManifest(
            zone_id=zone_id,
            engine="faiss+sqlite",
            embedding_model=self.embedding_model,
            embedding_dim=self.embedding_dim,
            distance=self.distance,
            normalized=True,
            chunk_strategy=self.chunk_strategy,
            build_time=utc_now_iso(),
            record_count=counts["knowledge"],
            baseline_count=counts["baseline"],
        )
        manifest_path.write_text(
            json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self._write_checksums(
            [
                sqlite_out_path,
                index_out_path,
                baseline_vec_path,
                baseline_meta_path,
                manifest_path,
            ],
            checksums_path,
        )

        return {
            "zone_id": zone_id,
            "target_dir": str(zone_dir),
            "sqlite_file": str(sqlite_out_path),
            "faiss_file": str(index_out_path),
            "manifest_file": str(manifest_path),
            "checksums_file": str(checksums_path),
        }

    def _fetch_rows_by_ids(
        self,
        zone_id: ZoneId,
        candidate_ids: list[int],
        filters: dict[str, Any] | None = None,
    ) -> list[sqlite3.Row]:
        placeholders = ",".join(["?"] * len(candidate_ids))
        params: list[Any] = [zone_id, *candidate_ids]
        sql = f"""
            SELECT id, chunk_id, doc_id, zone_id, record_type, text, filename, source_path, extension,
                   chunk_index, chunk_count, char_len, content_hash, mtime, size_bytes, file_type, language
            FROM chunks
            WHERE zone_id = ? AND is_deleted = 0 AND record_type = 'knowledge'
              AND id IN ({placeholders})
        """
        for key, value in (filters or {}).items():
            if key not in self._FILTERABLE_FIELDS:
                continue
            if isinstance(value, (str, int, float, bool)):
                sql += f" AND {key} = ?"
                params.append(value)

        with self._connect() as conn:
            return conn.execute(sql, params).fetchall()

    def _rebuild_zone_index(self, zone_id: ZoneId) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, v.vector
                FROM chunks c
                JOIN chunk_vectors v ON c.chunk_id = v.chunk_id
                WHERE c.zone_id = ? AND c.is_deleted = 0 AND c.record_type = 'knowledge'
                ORDER BY c.id ASC
                """,
                (zone_id,),
            ).fetchall()

        index = self._new_index()
        if rows:
            ids = np.asarray([int(r["id"]) for r in rows], dtype=np.int64)
            vectors = np.vstack(
                [self._normalize(np.frombuffer(r["vector"], dtype=np.float32)) for r in rows]
            ).astype(np.float32)
            index.add_with_ids(vectors, ids)

        faiss.write_index(index, str(self._zone_index_path(zone_id)))
        self._indexes[zone_id] = index

    def _load_zone_index(self, zone_id: ZoneId) -> Any:
        cached = self._indexes.get(zone_id)
        if cached is not None:
            return cached

        path = self._zone_index_path(zone_id)
        if path.exists():
            index = faiss.read_index(str(path))
        else:
            index = self._new_index()
            faiss.write_index(index, str(path))
        self._indexes[zone_id] = index
        return index

    def _new_index(self) -> Any:
        # 第一阶段使用 FlatIP，优先保证行为可预期。
        # 后续可切换到 IVF-PQ，且不需要改动对外接口。
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.embedding_dim))

    def _zone_index_path(self, zone_id: ZoneId) -> Path:
        return self.index_dir / f"{zone_id}.faiss"

    def _zone_counts(self, zone_id: ZoneId) -> dict[str, int]:
        with self._connect() as conn:
            knowledge = conn.execute(
                """
                SELECT COUNT(1)
                FROM chunks
                WHERE zone_id = ? AND is_deleted = 0 AND record_type = 'knowledge'
                """,
                (zone_id,),
            ).fetchone()[0]
            baseline = conn.execute(
                """
                SELECT COUNT(1)
                FROM chunks
                WHERE zone_id = ? AND is_deleted = 0 AND record_type = 'baseline'
                """,
                (zone_id,),
            ).fetchone()[0]
        return {"knowledge": int(knowledge), "baseline": int(baseline)}

    def _export_zone_sqlite(self, zone_id: ZoneId, sqlite_out_path: Path) -> None:
        with self._connect() as src_conn:
            zone_row = src_conn.execute(
                "SELECT zone_id, embedding_dim, created_at FROM zones WHERE zone_id = ?",
                (zone_id,),
            ).fetchone()
            chunk_rows = src_conn.execute(
                "SELECT * FROM chunks WHERE zone_id = ? AND is_deleted = 0 ORDER BY id ASC",
                (zone_id,),
            ).fetchall()
            vector_rows = src_conn.execute(
                """
                SELECT v.chunk_id, v.zone_id, v.vector, v.dim, v.updated_at
                FROM chunk_vectors v
                JOIN chunks c ON c.chunk_id = v.chunk_id
                WHERE v.zone_id = ? AND c.is_deleted = 0
                """,
                (zone_id,),
            ).fetchall()

        with sqlite3.connect(sqlite_out_path) as dst_conn:
            dst_conn.row_factory = sqlite3.Row
            dst_conn.executescript(self._schema_sql())
            if zone_row:
                dst_conn.execute(
                    "INSERT OR REPLACE INTO zones (zone_id, embedding_dim, created_at) VALUES (?, ?, ?)",
                    (zone_row["zone_id"], zone_row["embedding_dim"], zone_row["created_at"]),
                )

            chunk_cols = [
                "id",
                "chunk_id",
                "doc_id",
                "zone_id",
                "record_type",
                "text",
                "filename",
                "source_path",
                "extension",
                "chunk_index",
                "chunk_count",
                "char_len",
                "content_hash",
                "mtime",
                "size_bytes",
                "file_type",
                "language",
                "is_deleted",
                "created_at",
                "updated_at",
            ]
            chunk_sql = f"""
                INSERT INTO chunks ({", ".join(chunk_cols)})
                VALUES ({", ".join(["?"] * len(chunk_cols))})
            """
            for row in chunk_rows:
                dst_conn.execute(chunk_sql, tuple(row[c] for c in chunk_cols))

            vec_sql = """
                INSERT OR REPLACE INTO chunk_vectors (chunk_id, zone_id, vector, dim, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """
            for row in vector_rows:
                dst_conn.execute(
                    vec_sql,
                    (
                        row["chunk_id"],
                        row["zone_id"],
                        row["vector"],
                        row["dim"],
                        row["updated_at"],
                    ),
                )
            dst_conn.commit()

    def _write_checksums(self, files: list[Path], output_path: Path) -> None:
        checksums: dict[str, str] = {}
        for file_path in files:
            checksums[file_path.name] = self._sha256(file_path)
        output_path.write_text(
            json.dumps(checksums, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _sha256(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            while True:
                block = f.read(1024 * 1024)
                if not block:
                    break
                hasher.update(block)
        return hasher.hexdigest()

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        if self.distance != "cosine":
            return vector.astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector.astype(np.float32)
        return (vector / norm).astype(np.float32)

    def _vector_to_blob(self, vector: list[float]) -> bytes:
        arr = np.asarray(vector, dtype=np.float32)
        if self.distance == "cosine":
            arr = self._normalize(arr)
        return arr.astype(np.float32).tobytes()

    def _blob_to_vector(self, blob: bytes) -> list[float]:
        return np.frombuffer(blob, dtype=np.float32).tolist()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _schema_sql(self) -> str:
        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "sqlite_schema.sql"
        return schema_path.read_text(encoding="utf-8")

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(self._schema_sql())
            conn.commit()
