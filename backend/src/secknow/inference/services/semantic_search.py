from __future__ import annotations

from typing import Any

from secknow.vector_store.services.vector_service import VectorInfrastructureService

from ..retrieval.semantic import SemanticRetriever
from ..schemas.result import SearchResult
from .formatter import SearchResultFormatter
from .query_encoder import QueryEncoder
from .reranker import SearchReranker


class SemanticSearchService:
    """4.4 语义检索主流程。"""

    def __init__(self, vector_service: VectorInfrastructureService) -> None:
        self.vector_service = vector_service
        self.encoder = QueryEncoder()
        self.retriever = SemanticRetriever(vector_service=vector_service)
        self.reranker = SearchReranker()
        self.formatter = SearchResultFormatter()

    def search(
        self,
        *,
        query: str,
        zone_id: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        query_vector = self.encoder.encode(query)
        hits = self.retriever.hybrid_search(
            zone_id=zone_id,
            query=query,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )
        ranked = self.reranker.rerank(query=query, hits=hits)
        return self.formatter.format(ranked)
