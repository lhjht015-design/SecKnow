from __future__ import annotations

from ..models import RankedSearchResult
from ..schemas.result import SearchResult


class SearchResultFormatter:
    """Convert retrieval hits into stable 4.4 result schema."""

    def format(self, ranked_results: list[RankedSearchResult]) -> list[SearchResult]:
        formatted: list[SearchResult] = []
        for item in ranked_results:
            metadata = item.hit.metadata
            formatted.append(
                SearchResult(
                    chunk_id=item.hit.chunk_id,
                    doc_id=item.hit.doc_id,
                    zone_id=item.hit.zone_id,
                    score=item.rerank_score,
                    text=item.hit.text,
                    filename=metadata.get("filename"),
                    source_path=metadata.get("source_path"),
                    record_type=metadata.get("record_type"),
                )
            )
        return formatted
