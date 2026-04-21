from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from secknow.vector_store.models import SearchHit


@dataclass(slots=True)
class RetrievalContext:
    query: str
    query_vector: list[float]
    zone_id: str
    top_k: int
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RankedSearchResult:
    hit: SearchHit
    rerank_score: float
    matched_terms: list[str] = field(default_factory=list)
