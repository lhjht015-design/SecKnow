from __future__ import annotations

from typing import Any

from secknow.vector_store.models import SearchHit
from secknow.vector_store.services.vector_service import VectorInfrastructureService


class SemanticRetriever:
    """Thin 4.4 adapter over 4.3 retrieval methods."""

    def __init__(self, vector_service: VectorInfrastructureService) -> None:
        self.vector_service = vector_service

    def hybrid_search(
        self,
        *,
        zone_id: str,
        query: str,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        return self.vector_service.hybrid_search(
            zone_id=zone_id,
            query=query,
            query_vec=query_vector,
            top_k=top_k,
            filters=filters,
        )
