from __future__ import annotations

from typing import Any

from secknow.vector_store.models import SearchHit, ZoneId
from secknow.vector_store.services.sparse import SparseTextIndex
from secknow.vector_store.stores.base import VectorStore


class HybridRetriever:
    """与后端无关的 dense+sparse 融合层。"""

    def __init__(
        self,
        dense_store: VectorStore,
        sparse_index: SparseTextIndex,
        rrf_k: int = 60,
    ) -> None:
        self.dense_store = dense_store
        self.sparse_index = sparse_index
        self.rrf_k = rrf_k

    def hybrid_search(
        self,
        zone_id: ZoneId,
        query: str,
        query_vec: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        dense_hits = self.dense_store.search(
            zone_id=zone_id,
            query_vec=query_vec,
            top_k=max(top_k * 5, top_k),
            filters=filters,
        )
        sparse_hits = self.sparse_index.search(
            zone_id=zone_id,
            query=query,
            top_k=max(top_k * 5, top_k),
        )
        return self._rrf_fuse(dense_hits, sparse_hits, top_k=top_k)

    def _rrf_fuse(
        self,
        dense_hits: list[SearchHit],
        sparse_hits: list[Any],
        top_k: int,
    ) -> list[SearchHit]:
        scored: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            score = 1.0 / (self.rrf_k + rank)
            bucket = scored.setdefault(hit.chunk_id, {"hit": hit, "score": 0.0})
            bucket["score"] += score

        for rank, hit in enumerate(sparse_hits, start=1):
            score = 1.0 / (self.rrf_k + rank)
            if hit.chunk_id not in scored:
                scored[hit.chunk_id] = {
                    "hit": SearchHit(
                        chunk_id=hit.chunk_id,
                        doc_id=hit.doc_id,
                        zone_id=hit.zone_id,
                        score=0.0,
                        text=hit.text,
                        metadata=hit.metadata,
                    ),
                    "score": 0.0,
                }
            scored[hit.chunk_id]["score"] += score

        fused = []
        for entry in scored.values():
            base_hit: SearchHit = entry["hit"]
            fused.append(
                SearchHit(
                    chunk_id=base_hit.chunk_id,
                    doc_id=base_hit.doc_id,
                    zone_id=base_hit.zone_id,
                    score=float(entry["score"]),
                    text=base_hit.text,
                    metadata=base_hit.metadata,
                )
            )
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[:top_k]
