from __future__ import annotations

from secknow.vector_store.models import SearchHit

from ..config import DEFAULT_DENSE_WEIGHT, DEFAULT_KEYWORD_WEIGHT
from ..models import RankedSearchResult


class SearchReranker:
    """Small local reranker for 4.4 search output."""

    def rerank(self, query: str, hits: list[SearchHit]) -> list[RankedSearchResult]:
        terms = [term for term in query.lower().split() if term]
        ranked: list[RankedSearchResult] = []
        for hit in hits:
            text = hit.text.lower()
            matched_terms = [term for term in terms if term in text]
            keyword_score = len(matched_terms) / max(len(terms), 1)
            score = (
                DEFAULT_DENSE_WEIGHT * hit.score
                + DEFAULT_KEYWORD_WEIGHT * keyword_score
            )
            ranked.append(
                RankedSearchResult(
                    hit=hit,
                    rerank_score=score,
                    matched_terms=matched_terms,
                )
            )
        ranked.sort(key=lambda item: item.rerank_score, reverse=True)
        return ranked
