from __future__ import annotations

from typing import Any

from secknow.vector_store.services.vector_service import VectorInfrastructureService

from .config import DEFAULT_TOP_K, DEFAULT_ZONE_ID
from .schemas.result import CodeRiskResult, SearchResult
from .services.code_risk_scan import CodeRiskScanService
from .services.semantic_search import SemanticSearchService


class InferenceService:
    """给 4.5 和后续应用层使用的 4.4 统一门面。"""

    def __init__(self, vector_service: VectorInfrastructureService) -> None:
        self.vector_service = vector_service
        self.semantic = SemanticSearchService(vector_service=vector_service)
        self.code_scan = CodeRiskScanService(vector_service=vector_service)

    def semantic_search(
        self,
        query: str,
        zone_id: str = DEFAULT_ZONE_ID,
        top_k: int = DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        return self.semantic.search(
            query=query,
            zone_id=zone_id,
            top_k=top_k,
            filters=filters,
        )

    def code_risk_scan(
        self,
        code_snippet: str,
        lang: str,
        zone_id: str = DEFAULT_ZONE_ID,
        top_k: int = DEFAULT_TOP_K,
    ) -> CodeRiskResult:
        return self.code_scan.scan(
            code_snippet=code_snippet,
            lang=lang,
            zone_id=zone_id,
            top_k=top_k,
        )


def semantic_search(
    vector_service: VectorInfrastructureService,
    query: str,
    zone_id: str = DEFAULT_ZONE_ID,
    top_k: int = DEFAULT_TOP_K,
    filters: dict[str, Any] | None = None,
) -> list[SearchResult]:
    service = InferenceService(vector_service=vector_service)
    return service.semantic_search(
        query=query,
        zone_id=zone_id,
        top_k=top_k,
        filters=filters,
    )


def code_risk_scan(
    vector_service: VectorInfrastructureService,
    code_snippet: str,
    lang: str,
    zone_id: str = DEFAULT_ZONE_ID,
    top_k: int = DEFAULT_TOP_K,
) -> CodeRiskResult:
    service = InferenceService(vector_service=vector_service)
    return service.code_risk_scan(
        code_snippet=code_snippet,
        lang=lang,
        zone_id=zone_id,
        top_k=top_k,
    )
