from __future__ import annotations

from typing import Any

from secknow.vector_store.models import (
    BaselineBundle,
    ChunkRecord,
    DeleteResult,
    SearchHit,
    UpsertResult,
    ZoneId,
)
from secknow.vector_store.services.hybrid import HybridRetriever
from secknow.vector_store.services.sparse import SparseTextIndex
from secknow.vector_store.stores.base import VectorStore


class VectorInfrastructureService:
    """统一的 4.3 服务门面，供 4.5 API 层与 4.4 推理层调用。"""

    def __init__(self, dense_store: VectorStore, sparse_index: SparseTextIndex) -> None:
        self.dense_store = dense_store
        self.sparse_index = sparse_index
        self.hybrid = HybridRetriever(
            dense_store=dense_store, sparse_index=sparse_index
        )

    def ensure_zone(self, zone_id: ZoneId, dim: int) -> None:
        self.dense_store.ensure_zone(zone_id=zone_id, dim=dim)

    def upsert(self, zone_id: ZoneId, records: list[ChunkRecord]) -> UpsertResult:
        result = self.dense_store.upsert(zone_id=zone_id, records=records)
        self.sparse_index.upsert(zone_id=zone_id, records=records)
        return result

    def search(
        self,
        zone_id: ZoneId,
        query_vec: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        if top_k <= 0:
            return []
        return self.dense_store.search(
            zone_id=zone_id,
            query_vec=query_vec,
            top_k=top_k,
            filters=filters,
        )

    def hybrid_search(
        self,
        zone_id: ZoneId,
        query: str,
        query_vec: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        if top_k <= 0:
            return []
        return self.hybrid.hybrid_search(
            zone_id=zone_id,
            query=query,
            query_vec=query_vec,
            top_k=top_k,
            filters=filters,
        )

    def delete(self, zone_id: ZoneId, chunk_ids: list[str]) -> DeleteResult:
        result = self.dense_store.delete(zone_id=zone_id, chunk_ids=chunk_ids)
        self.sparse_index.delete(zone_id=zone_id, chunk_ids=chunk_ids)
        return result

    def delete_by_doc_id(self, zone_id: ZoneId, doc_id: str) -> DeleteResult:
        result = self.dense_store.delete_by_doc_id(zone_id=zone_id, doc_id=doc_id)
        self.sparse_index.delete_by_doc_id(zone_id=zone_id, doc_id=doc_id)
        return result

    def export_zone(self, zone_id: ZoneId, target_dir: str) -> dict[str, Any]:
        return self.dense_store.export_zone(zone_id=zone_id, target_dir=target_dir)

    def get_baseline(self, zone_id: ZoneId) -> BaselineBundle:
        return self.dense_store.get_baseline(zone_id=zone_id)
