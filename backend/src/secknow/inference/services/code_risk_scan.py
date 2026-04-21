from __future__ import annotations

from secknow.vector_store.services.vector_service import VectorInfrastructureService

from ..config import SUPPORTED_CODE_LANGUAGES
from ..exceptions import UnsupportedLanguageError
from ..retrieval.semantic import SemanticRetriever
from ..scanners.router import build_code_scanner
from ..schemas.result import CodeRiskItem, CodeRiskResult
from .formatter import SearchResultFormatter
from .query_encoder import QueryEncoder
from .reranker import SearchReranker


class CodeRiskScanService:
    """4.4 代码风险扫描主流程骨架。"""

    def __init__(self, vector_service: VectorInfrastructureService) -> None:
        self.vector_service = vector_service
        self.encoder = QueryEncoder()
        self.retriever = SemanticRetriever(vector_service=vector_service)
        self.reranker = SearchReranker()
        self.formatter = SearchResultFormatter()

    def scan(
        self,
        *,
        code_snippet: str,
        lang: str,
        zone_id: str,
        top_k: int,
    ) -> CodeRiskResult:
        normalized_lang = lang.strip().lower()
        if normalized_lang not in SUPPORTED_CODE_LANGUAGES:
            raise UnsupportedLanguageError(
                f"unsupported scan language: {normalized_lang}"
            )

        scanner = build_code_scanner(normalized_lang)
        units = scanner.extract_units(code_snippet, normalized_lang)
        items: list[CodeRiskItem] = []

        for unit in units[:top_k]:
            query_vector = self.encoder.encode(unit)
            hits = self.retriever.hybrid_search(
                zone_id=zone_id,
                query=unit,
                query_vector=query_vector,
                top_k=top_k,
                filters={"file_type": "code"},
            )
            ranked = self.reranker.rerank(query=unit, hits=hits)
            references = self.formatter.format(ranked)
            items.append(
                CodeRiskItem(
                    code_unit=unit,
                    summary="Potentially relevant code risk references found.",
                    severity="info",
                    references=references,
                )
            )

        return CodeRiskResult(
            lang=normalized_lang,
            zone_id=zone_id,
            items=items,
        )
