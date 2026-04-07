from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

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


class QdrantVectorStore(VectorStore):
    """基于 Qdrant 的在线向量存储后端。"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: str | None = None,
        https: bool = False,
        collection_prefix: str = "knowledge",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        distance: str = "cosine",
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qm
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is required for QdrantVectorStore. "
                "Install with: pip install qdrant-client"
            ) from exc

        self.client = QdrantClient(host=host, port=port, api_key=api_key, https=https)
        self.qm = qm
        self.collection_prefix = collection_prefix
        self.embedding_model = embedding_model
        self.distance = distance.lower()
        self.chunk_strategy = {"type": "hybrid", "max_tokens": 500, "overlap": 50}

    def ensure_zone(self, zone_id: ZoneId, dim: int) -> None:
        assert_zone(zone_id)
        collection_name = self._collection_name(zone_id)
        if self._collection_exists(collection_name):
            return

        distance = self._resolve_distance()
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=self.qm.VectorParams(size=dim, distance=distance),
        )
        self._ensure_payload_indexes(collection_name)

    def upsert(self, zone_id: ZoneId, records: list[ChunkRecord]) -> UpsertResult:
        assert_zone(zone_id)
        if not records:
            return UpsertResult(zone_id=zone_id, attempted=0, inserted=0, chunk_ids=[])

        dim = len(records[0].vector)
        self.ensure_zone(zone_id, dim)

        points = []
        chunk_ids: list[str] = []
        for record in records:
            metadata = record.metadata
            if metadata.zone_id != zone_id:
                raise ValueError(
                    f"Record zone mismatch: metadata.zone_id={metadata.zone_id}, upsert zone={zone_id}"
                )
            if len(record.vector) != dim:
                raise ValueError("Embedding dim mismatch inside upsert batch.")

            chunk_id = metadata.chunk_id or generate_chunk_id(metadata)
            metadata.chunk_id = chunk_id
            chunk_ids.append(chunk_id)
            payload = metadata.model_dump()
            payload["text"] = record.text

            points.append(
                self.qm.PointStruct(
                    id=self._point_id(chunk_id),
                    vector=record.vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self._collection_name(zone_id),
            points=points,
            wait=True,
        )
        return UpsertResult(
            zone_id=zone_id,
            attempted=len(records),
            inserted=len(points),
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
        query_filter = self._build_query_filter(filters)
        response = self.client.query_points(
            collection_name=self._collection_name(zone_id),
            query=query_vec,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(response, "points", response)

        hits: list[SearchHit] = []
        for point in points:
            payload = dict(point.payload or {})
            text = str(payload.pop("text", ""))
            chunk_id = str(payload.get("chunk_id", point.id))
            doc_id = str(payload.get("doc_id", ""))
            score = float(getattr(point, "score", 0.0))
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    zone_id=zone_id,
                    score=score,
                    text=text,
                    metadata=payload,
                )
            )
        return hits

    def delete(self, zone_id: ZoneId, chunk_ids: list[str]) -> DeleteResult:
        assert_zone(zone_id)
        if not chunk_ids:
            return DeleteResult(zone_id=zone_id, requested=0, deleted=0, chunk_ids=[])

        self.client.delete(
            collection_name=self._collection_name(zone_id),
            points_selector=self.qm.PointIdsList(
                points=[self._point_id(chunk_id) for chunk_id in chunk_ids]
            ),
            wait=True,
        )
        return DeleteResult(
            zone_id=zone_id,
            requested=len(chunk_ids),
            deleted=len(chunk_ids),
            chunk_ids=chunk_ids,
        )

    def get_baseline(self, zone_id: ZoneId) -> BaselineBundle:
        assert_zone(zone_id)
        filter_obj = self.qm.Filter(
            must=[
                self.qm.FieldCondition(
                    key="record_type", match=self.qm.MatchValue(value="baseline")
                )
            ]
        )

        vectors: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self._collection_name(zone_id),
                scroll_filter=filter_obj,
                with_payload=True,
                with_vectors=True,
                limit=256,
                offset=offset,
            )
            for point in points:
                raw_vector = point.vector
                if isinstance(raw_vector, dict):
                    raw_vector = next(iter(raw_vector.values()), [])
                vectors.append(list(raw_vector or []))
                payload = dict(point.payload or {})
                payload.pop("text", None)
                metadatas.append(payload)
            if offset is None:
                break
        return BaselineBundle(zone_id=zone_id, vectors=vectors, metadatas=metadatas)

    def export_zone(self, zone_id: ZoneId, target_dir: str) -> dict[str, Any]:
        assert_zone(zone_id)
        export_root = Path(target_dir).expanduser().resolve()
        zone_dir = export_root / zone_id
        zone_dir.mkdir(parents=True, exist_ok=True)

        records_path = zone_dir / "qdrant_export.jsonl"
        manifest_path = zone_dir / "manifest.json"

        records: list[dict[str, Any]] = []
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self._collection_name(zone_id),
                with_payload=True,
                with_vectors=True,
                limit=512,
                offset=offset,
            )
            for point in points:
                payload = dict(point.payload or {})
                records.append(
                    {
                        "id": str(point.id),
                        "vector": point.vector,
                        "payload": payload,
                    }
                )
            if offset is None:
                break

        with records_path.open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        baseline_count = sum(
            1 for row in records if row.get("payload", {}).get("record_type") == "baseline"
        )
        manifest = ExportManifest(
            zone_id=zone_id,
            engine="qdrant",
            embedding_model=self.embedding_model,
            embedding_dim=self._extract_embedding_dim(records),
            distance=self.distance,
            normalized=True,
            chunk_strategy=self.chunk_strategy,
            build_time=utc_now_iso(),
            record_count=len(records),
            baseline_count=baseline_count,
        )
        manifest_path.write_text(
            json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "zone_id": zone_id,
            "target_dir": str(zone_dir),
            "records_file": str(records_path),
            "manifest_file": str(manifest_path),
            "record_count": len(records),
        }

    def _collection_name(self, zone_id: ZoneId) -> str:
        return f"{self.collection_prefix}_{zone_id}"

    def _collection_exists(self, collection_name: str) -> bool:
        if hasattr(self.client, "collection_exists"):
            return bool(self.client.collection_exists(collection_name))
        try:
            self.client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False

    def _resolve_distance(self) -> Any:
        mapping = {
            "cosine": self.qm.Distance.COSINE,
            "dot": self.qm.Distance.DOT,
            "euclid": self.qm.Distance.EUCLID,
        }
        if self.distance not in mapping:
            raise ValueError("distance must be one of: cosine, dot, euclid")
        return mapping[self.distance]

    def _point_id(self, chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def _build_query_filter(self, filters: dict[str, Any] | None) -> Any:
        conditions = []
        if not filters or "record_type" not in filters:
            conditions.append(
                self.qm.FieldCondition(
                    key="record_type", match=self.qm.MatchValue(value="knowledge")
                )
            )
        for key, value in (filters or {}).items():
            if isinstance(value, (str, int, float, bool)):
                conditions.append(
                    self.qm.FieldCondition(
                        key=key,
                        match=self.qm.MatchValue(value=value),
                    )
                )
        if not conditions:
            return None
        return self.qm.Filter(must=conditions)

    def _extract_embedding_dim(self, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        first = records[0].get("vector")
        if isinstance(first, dict):
            first = next(iter(first.values()), [])
        return len(first or [])

    def _ensure_payload_indexes(self, collection_name: str) -> None:
        """为常用过滤字段尽力创建 payload 索引。"""
        fields = [
            ("doc_id", self.qm.PayloadSchemaType.KEYWORD),
            ("record_type", self.qm.PayloadSchemaType.KEYWORD),
            ("filename", self.qm.PayloadSchemaType.KEYWORD),
            ("mtime", self.qm.PayloadSchemaType.INTEGER),
        ]
        for field_name, schema_type in fields:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception:
                # 不同 qdrant 版本的幂等行为有差异；若已存在则忽略。
                continue
