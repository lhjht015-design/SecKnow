from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from secknow.domain.models import Chunk, CodeSymbol, ParsedElement, SourceDocument

SCHEMA_VERSION = 1

# FTS5 的 unicode61 分词器不认识中文：一整段连续汉字会被切成**一个** token，
# 于是「报错注入」搜不到正文里的「数据库报错注入的方法」——这是中文知识库最常见的召回塌陷。
# 解决办法是在写入与查询两侧都用 jieba 的搜索引擎模式切一遍，
# 它除了词本身还会吐出子词（数据/据库/数据库），把这些召回补回来。
# 英文不受影响：jieba 会把英文单词原样保留。
_FTS_STOPWORDS = frozenset({
    "的", "了", "和", "与", "或", "在", "是", "有", "为", "对", "从", "到", "中", "上", "下",
    "这个", "那个", "可以", "我们", "你们", "他们", "一个", "什么", "怎么", "如何", "以及",
})
_segment_cache: dict[str, str] = {}


def segment_for_fts(text: str) -> str:
    """把文本切成空格分隔的 token 序列，供 FTS5 索引与查询共用。"""
    if not text:
        return ""
    cached = _segment_cache.get(text)
    if cached is not None:
        return cached
    import jieba

    tokens: list[str] = []
    for token in jieba.cut_for_search(text):
        stripped = token.strip()
        if not stripped or stripped in _FTS_STOPWORDS:
            continue
        # 纯标点没有检索价值，还会把 token 数撑大
        if not any(ch.isalnum() for ch in stripped):
            continue
        tokens.append(stripped.casefold())
    result = " ".join(tokens)
    # 缓存上限，避免导入几万块时把内存吃满
    if len(_segment_cache) < 200_000:
        _segment_cache[text] = result
    return result


class JobLeaseLostError(RuntimeError):
    pass


class MetadataStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        connection = self.connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER NOT NULL
                );
                INSERT INTO schema_info(version)
                SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_info);

                CREATE TABLE IF NOT EXISTS documents (
                    source_id TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    absolute_path TEXT NOT NULL,
                    format TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    language TEXT,
                    modified_at TEXT,
                    discovered_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(source_id, version_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_active
                    ON documents(source_id) WHERE active = 1;

                CREATE TABLE IF NOT EXISTS elements (
                    element_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    source_version TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    element_json TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_elements_source
                    ON elements(source_id, source_version, position);

                CREATE TABLE IF NOT EXISTS code_symbols (
                    symbol_id TEXT PRIMARY KEY,
                    element_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_version TEXT NOT NULL,
                    symbol_json TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_code_symbols_source
                    ON code_symbols(source_id, source_version);

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    source_version TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    text TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    citation_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    chunk_json TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_source
                    ON chunks(source_id, source_version, position);
                CREATE INDEX IF NOT EXISTS idx_chunks_active ON chunks(active);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    title,
                    section,
                    text,
                    tokenize='unicode61 remove_diacritics 2'
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    stage TEXT NOT NULL DEFAULT 'queued',
                    error TEXT,
                    owner TEXT,
                    lease_until TEXT,
                    heartbeat_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_queue
                    ON jobs(status, created_at);

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    status TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    parent_snapshot_id TEXT,
                    created_at TEXT NOT NULL,
                    published_at TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshots_published
                    ON snapshots(source_id) WHERE status = 'published';
                """
            )
            version = connection.execute("SELECT version FROM schema_info").fetchone()[0]
            if version != SCHEMA_VERSION:
                raise RuntimeError(f"Unsupported metadata schema version: {version}")
        finally:
            connection.close()

    def upsert_document(self, document: SourceDocument, active: bool = False) -> None:
        values = document.model_dump(mode="json")
        with self.transaction() as db:
            db.execute(
                """
                INSERT INTO documents(
                    source_id, version_id, title, relative_path, absolute_path,
                    format, media_type, size_bytes, sha256, language, modified_at,
                    discovered_at, metadata_json, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, version_id) DO UPDATE SET
                    title=excluded.title,
                    relative_path=excluded.relative_path,
                    absolute_path=excluded.absolute_path,
                    metadata_json=excluded.metadata_json
                """,
                (
                    document.source_id,
                    document.version_id,
                    document.title,
                    document.relative_path,
                    document.absolute_path,
                    document.format.value,
                    document.media_type,
                    document.size_bytes,
                    document.sha256,
                    document.language,
                    values["modified_at"],
                    values["discovered_at"],
                    json.dumps(document.metadata, ensure_ascii=False),
                    int(active),
                ),
            )

    def replace_elements(
        self,
        source_id: str,
        version_id: str,
        elements: Sequence[ParsedElement],
    ) -> None:
        with self.transaction() as db:
            db.execute(
                "DELETE FROM elements WHERE source_id=? AND source_version=?",
                (source_id, version_id),
            )
            db.executemany(
                """
                INSERT INTO elements(
                    element_id, source_id, source_version, position, element_json, active
                ) VALUES (?, ?, ?, ?, ?, 0)
                """,
                [
                    (
                        element.element_id,
                        element.source_id,
                        element.source_version,
                        element.order,
                        element.model_dump_json(),
                    )
                    for element in elements
                ],
            )

    def replace_code_symbols(
        self,
        source_id: str,
        version_id: str,
        symbols: Sequence[CodeSymbol],
    ) -> None:
        with self.transaction() as db:
            db.execute(
                "DELETE FROM code_symbols WHERE source_id=? AND source_version=?",
                (source_id, version_id),
            )
            db.executemany(
                """
                INSERT INTO code_symbols(
                    symbol_id, element_id, source_id, source_version, symbol_json, active
                ) VALUES (?, ?, ?, ?, ?, 0)
                """,
                [
                    (
                        symbol.symbol_id,
                        symbol.element_id,
                        source_id,
                        version_id,
                        symbol.model_dump_json(),
                    )
                    for symbol in symbols
                ],
            )

    def replace_chunks(self, source_id: str, version_id: str, chunks: Sequence[Chunk]) -> None:
        with self.transaction() as db:
            db.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN "
                "(SELECT chunk_id FROM chunks WHERE source_id=? AND source_version=?)",
                (source_id, version_id),
            )
            db.execute(
                "DELETE FROM chunks WHERE source_id=? AND source_version=?",
                (source_id, version_id),
            )
            for chunk in chunks:
                db.execute(
                    """
                    INSERT INTO chunks(
                        chunk_id, source_id, source_version, content_sha256,
                        text, position, citation_json, metadata_json, chunk_json, active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.source_id,
                        chunk.source_version,
                        chunk.content_sha256,
                        chunk.text,
                        chunk.order,
                        chunk.citation.model_dump_json(),
                        json.dumps(chunk.metadata, ensure_ascii=False),
                        chunk.model_dump_json(),
                    ),
                )
                db.execute(
                    "INSERT INTO chunks_fts(chunk_id, title, section, text) VALUES (?, ?, ?, ?)",
                    (
                        chunk.chunk_id,
                        segment_for_fts(chunk.citation.title),
                        segment_for_fts(" / ".join(chunk.section_path)),
                        segment_for_fts(chunk.text),
                    ),
                )

    def publish_version(
        self,
        source_id: str,
        version_id: str,
        tfidf_version: str,
        graph_version: str,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self.transaction() as db:
            exists = db.execute(
                "SELECT 1 FROM documents WHERE source_id=? AND version_id=?",
                (source_id, version_id),
            ).fetchone()
            if not exists:
                raise KeyError(f"Unknown document version: {source_id}/{version_id}")
            db.execute("UPDATE documents SET active=0 WHERE source_id=?", (source_id,))
            db.execute(
                "UPDATE documents SET active=1 WHERE source_id=? AND version_id=?",
                (source_id, version_id),
            )
            db.execute("UPDATE elements SET active=0 WHERE source_id=?", (source_id,))
            db.execute(
                "UPDATE elements SET active=1 WHERE source_id=? AND source_version=?",
                (source_id, version_id),
            )
            db.execute("UPDATE code_symbols SET active=0 WHERE source_id=?", (source_id,))
            db.execute(
                "UPDATE code_symbols SET active=1 WHERE source_id=? AND source_version=?",
                (source_id, version_id),
            )
            db.execute("UPDATE chunks SET active=0 WHERE source_id=?", (source_id,))
            db.execute(
                "UPDATE chunks SET active=1 WHERE source_id=? AND source_version=?",
                (source_id, version_id),
            )
            settings = (
                ("active_tfidf_version", json.dumps(tfidf_version), now),
                ("active_graph_version", json.dumps(graph_version), now),
            )
            db.executemany(
                """
                INSERT INTO settings(key, value_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json, updated_at=excluded.updated_at
                """,
                settings,
            )

    def activate_version(self, source_id: str, version_id: str) -> None:
        with self.transaction() as db:
            exists = db.execute(
                "SELECT 1 FROM documents WHERE source_id=? AND version_id=?",
                (source_id, version_id),
            ).fetchone()
            if not exists:
                raise KeyError(f"Unknown document version: {source_id}/{version_id}")
            db.execute("UPDATE documents SET active=0 WHERE source_id=?", (source_id,))
            db.execute(
                "UPDATE documents SET active=1 WHERE source_id=? AND version_id=?",
                (source_id, version_id),
            )
            db.execute("UPDATE elements SET active=0 WHERE source_id=?", (source_id,))
            db.execute(
                "UPDATE elements SET active=1 WHERE source_id=? AND source_version=?",
                (source_id, version_id),
            )
            db.execute("UPDATE code_symbols SET active=0 WHERE source_id=?", (source_id,))
            db.execute(
                "UPDATE code_symbols SET active=1 WHERE source_id=? AND source_version=?",
                (source_id, version_id),
            )
            db.execute("UPDATE chunks SET active=0 WHERE source_id=?", (source_id,))
            db.execute(
                "UPDATE chunks SET active=1 WHERE source_id=? AND source_version=?",
                (source_id, version_id),
            )

    def active_code_symbols(
        self,
        excluding_source: str | None = None,
    ) -> tuple[CodeSymbol, ...]:
        query = "SELECT symbol_json FROM code_symbols WHERE active=1"
        parameters: list[Any] = []
        if excluding_source is not None:
            query += " AND source_id != ?"
            parameters.append(excluding_source)
        query += " ORDER BY source_id, symbol_id"
        with self.connect() as db:
            rows = db.execute(query, parameters).fetchall()
        return tuple(CodeSymbol.model_validate_json(row["symbol_json"]) for row in rows)

    def active_chunks(self, excluding_source: str | None = None) -> tuple[Chunk, ...]:
        query = "SELECT chunk_json FROM chunks WHERE active=1"
        parameters: list[Any] = []
        if excluding_source is not None:
            query += " AND source_id != ?"
            parameters.append(excluding_source)
        query += " ORDER BY source_id, position"
        with self.connect() as db:
            rows = db.execute(query, parameters).fetchall()
        return tuple(Chunk.model_validate_json(row["chunk_json"]) for row in rows)

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        with self.connect() as db:
            rows = db.execute(
                f"SELECT chunk_id, chunk_json FROM chunks WHERE active=1 "
                f"AND chunk_id IN ({placeholders})",
                list(chunk_ids),
            ).fetchall()
        return {row["chunk_id"]: Chunk.model_validate_json(row["chunk_json"]) for row in rows}

    def search_fts(self, query: str, top_k: int = 40) -> list[tuple[str, float]]:
        raw_terms = [term for term in query.replace('"', " ").split() if term]
        if not raw_terms:
            return []
        # 查询侧必须和索引侧用同一套切分，否则中文查询会整体失配
        tokens: list[str] = []
        for term in raw_terms:
            tokens.extend(segment_for_fts(term).split())
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return []
        expression = " OR ".join(f'"{token}"' for token in tokens)
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT f.chunk_id, bm25(chunks_fts, 0.0, 2.0, 1.0) AS score
                FROM chunks_fts AS f
                JOIN chunks AS c ON c.chunk_id=f.chunk_id
                WHERE chunks_fts MATCH ? AND c.active=1
                ORDER BY score
                LIMIT ?
                """,
                (expression, top_k),
            ).fetchall()
        return [(row["chunk_id"], float(-row["score"])) for row in rows]

    def rebuild_fts(self, batch_size: int = 2000) -> int:
        """用当前的分词规则重建整张 FTS 表。

        分词规则变了（例如从 unicode61 的整段中文改为 jieba 切分）之后，
        已有行的索引内容就作废了——`replace_chunks` 只会重写被重新导入的文档。
        本方法把全部块的 FTS 行按新规则重刷一遍，返回重刷的块数。
        """
        with self.connect() as db:
            rows = db.execute("SELECT chunk_id FROM chunks").fetchall()
        total = 0
        for start in range(0, len(rows), batch_size):
            batch = [row["chunk_id"] for row in rows[start:start + batch_size]]
            marks = ",".join("?" * len(batch))
            with self.transaction() as db:
                payload = db.execute(
                    f"SELECT chunk_id, citation_json, chunk_json, text "
                    f"FROM chunks WHERE chunk_id IN ({marks})",
                    batch,
                ).fetchall()
                db.execute(
                    f"DELETE FROM chunks_fts WHERE chunk_id IN ({marks})", batch
                )
                for row in payload:
                    citation = json.loads(row["citation_json"])
                    # section_path 只存在于 chunk_json 里，chunks 表没有单独的列
                    chunk_payload = json.loads(row["chunk_json"])
                    sections = chunk_payload.get("section_path") or []
                    db.execute(
                        "INSERT INTO chunks_fts(chunk_id, title, section, text)"
                        " VALUES (?, ?, ?, ?)",
                        (
                            row["chunk_id"],
                            segment_for_fts(str(citation.get("title", ""))),
                            segment_for_fts(" / ".join(str(s) for s in sections)),
                            segment_for_fts(str(row["text"])),
                        ),
                    )
            total += len(batch)
        return total

    def list_documents(self, active_only: bool = True) -> list[dict[str, Any]]:
        where = "WHERE active=1" if active_only else ""
        with self.connect() as db:
            rows = db.execute(f"SELECT * FROM documents {where} ORDER BY title").fetchall()
        return [dict(row) for row in rows]

    def document_page_audit(
        self,
        source_id: str,
        *,
        attention_only: bool = False,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any] | None:
        with self.connect() as db:
            document = db.execute(
                """
                SELECT * FROM documents
                WHERE source_id=?
                ORDER BY active DESC, discovered_at DESC, version_id DESC
                LIMIT 1
                """,
                (source_id,),
            ).fetchone()
            if document is None:
                return None
            rows = db.execute(
                """
                SELECT element_json
                FROM elements
                WHERE source_id=? AND source_version=?
                ORDER BY position
                """,
                (source_id, document["version_id"]),
            ).fetchall()

        counts = {name: 0 for name in ("native", "ocr", "ocr_required", "quarantine")}
        pages: list[dict[str, Any]] = []
        for row in rows:
            element = ParsedElement.model_validate_json(row["element_json"])
            if element.page is None:
                continue
            disposition = str(element.metadata.get("disposition", "native"))
            counts[disposition] = counts.get(disposition, 0) + 1
            pages.append(
                {
                    "page": element.page,
                    "disposition": disposition,
                    "indexable": disposition not in {"ocr_required", "quarantine"},
                    "parser": element.parser,
                    "extraction_confidence": element.extraction_confidence,
                    "diagnostics": list(dict.fromkeys(element.diagnostics)),
                }
            )

        filtered = (
            [page for page in pages if not page["indexable"]]
            if attention_only
            else pages
        )
        return {
            "source_id": source_id,
            "version_id": document["version_id"],
            "title": document["title"],
            "relative_path": document["relative_path"],
            "format": document["format"],
            "counts": {**counts, "total": len(pages)},
            "total_matching": len(filtered),
            "offset": offset,
            "limit": limit,
            "pages": filtered[offset : offset + limit],
        }

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def set_setting(self, key: str, value: Any) -> None:
        now = datetime.now(UTC).isoformat()
        with self.transaction() as db:
            db.execute(
                """
                INSERT INTO settings(key, value_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json, updated_at=excluded.updated_at
                """,
                (key, json.dumps(value, ensure_ascii=False), now),
            )

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self.connect() as db:
            row = db.execute("SELECT value_json FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row["value_json"]) if row else default

    def create_job(self, job_id: str, kind: str, payload: dict[str, Any]) -> str:
        now = datetime.now(UTC).isoformat()
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self.transaction() as db:
            rows = db.execute(
                """
                SELECT job_id, status, payload_json
                FROM jobs
                WHERE kind=? AND status IN ('queued', 'running')
                ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END,
                         created_at, job_id
                """,
                (kind,),
            ).fetchall()
            matching = [
                row
                for row in rows
                if json.loads(str(row["payload_json"])) == payload
            ]
            if matching:
                existing_id = str(matching[0]["job_id"])
                duplicate_ids = [
                    str(row["job_id"])
                    for row in matching[1:]
                    if row["status"] == "queued"
                ]
                if duplicate_ids:
                    placeholders = ",".join("?" for _ in duplicate_ids)
                    db.execute(
                        f"""
                        UPDATE jobs
                        SET status='superseded', stage='superseded',
                            owner=NULL, lease_until=NULL, updated_at=?
                        WHERE job_id IN ({placeholders}) AND status='queued'
                        """,
                        (now, *duplicate_ids),
                    )
                return existing_id
            db.execute(
                """
                INSERT INTO jobs(
                    job_id, kind, status, payload_json, created_at, updated_at
                ) VALUES (?, ?, 'queued', ?, ?, ?)
                """,
                (job_id, kind, payload_json, now, now),
            )
        return job_id

    def lease_job(self, owner: str, lease_seconds: int = 90) -> sqlite3.Row | None:
        now = datetime.now(UTC)
        lease_until = now + timedelta(seconds=lease_seconds)
        with self.transaction() as db:
            job = db.execute(
                """
                SELECT * FROM jobs
                WHERE status='queued'
                   OR (status='running' AND lease_until < ?)
                ORDER BY created_at
                LIMIT 1
                """,
                (now.isoformat(),),
            ).fetchone()
            if not job:
                return None
            db.execute(
                """
                UPDATE jobs
                SET status='running', owner=?, lease_until=?, heartbeat_at=?, updated_at=?
                WHERE job_id=?
                """,
                (owner, lease_until.isoformat(), now.isoformat(), now.isoformat(), job["job_id"]),
            )
            duplicate_rows = db.execute(
                """
                SELECT job_id, payload_json
                FROM jobs
                WHERE kind=? AND status='queued' AND job_id<>?
                """,
                (job["kind"], job["job_id"]),
            ).fetchall()
            payload = json.loads(str(job["payload_json"]))
            duplicate_ids = [
                str(row["job_id"])
                for row in duplicate_rows
                if json.loads(str(row["payload_json"])) == payload
            ]
            if duplicate_ids:
                placeholders = ",".join("?" for _ in duplicate_ids)
                db.execute(
                    f"""
                    UPDATE jobs
                    SET status='superseded', stage='superseded',
                        owner=NULL, lease_until=NULL, updated_at=?
                    WHERE job_id IN ({placeholders}) AND status='queued'
                    """,
                    (now.isoformat(), *duplicate_ids),
                )
            leased = db.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job["job_id"],)
            ).fetchone()
            return cast(sqlite3.Row | None, leased)

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress: float | None = None,
        error: str | None = None,
        owner: str | None = None,
        lease_seconds: int = 90,
    ) -> None:
        now = datetime.now(UTC)
        assignments: list[str] = ["updated_at=?"]
        values: list[Any] = [now.isoformat()]
        for column, value in (("status", status), ("stage", stage), ("progress", progress)):
            if value is not None:
                assignments.append(f"{column}=?")
                values.append(value)
        if error is not None:
            assignments.append("error=?")
            values.append(error)
        where = "job_id=?"
        if owner is not None:
            if status in {"queued", "succeeded", "failed"}:
                assignments.extend(("owner=NULL", "heartbeat_at=?", "lease_until=NULL"))
                values.append(now.isoformat())
            else:
                assignments.extend(("heartbeat_at=?", "lease_until=?"))
                values.extend(
                    (
                        now.isoformat(),
                        (now + timedelta(seconds=lease_seconds)).isoformat(),
                    )
                )
            where += " AND owner=?"
        values.append(job_id)
        if owner is not None:
            values.append(owner)
        with self.transaction() as db:
            cursor = db.execute(
                f"UPDATE jobs SET {', '.join(assignments)} WHERE {where}",
                values,
            )
            if owner is not None and cursor.rowcount != 1:
                raise JobLeaseLostError(f"Job lease lost: {job_id}")
