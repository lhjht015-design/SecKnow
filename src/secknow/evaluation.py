from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


def retrieval_metrics(
    relevant_ids: set[str],
    ranked_ids: list[str],
    k: int,
) -> RetrievalMetrics:
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant_ids:
        raise ValueError("relevant_ids cannot be empty")
    ranked = ranked_ids[:k]
    relevance = [1 if item_id in relevant_ids else 0 for item_id in ranked]
    hit_count = sum(relevance)
    reciprocal_rank = next(
        (1.0 / rank for rank, relevant in enumerate(relevance, 1) if relevant),
        0.0,
    )
    dcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(relevance, 1))
    ideal_hits = min(k, len(relevant_ids))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return RetrievalMetrics(
        precision_at_k=hit_count / k,
        recall_at_k=hit_count / len(relevant_ids),
        mrr=reciprocal_rank,
        ndcg_at_k=dcg / ideal_dcg if ideal_dcg else 0.0,
    )


def mean_metrics(rows: list[RetrievalMetrics]) -> RetrievalMetrics:
    if not rows:
        raise ValueError("at least one metric row is required")
    count = len(rows)
    return RetrievalMetrics(
        precision_at_k=sum(row.precision_at_k for row in rows) / count,
        recall_at_k=sum(row.recall_at_k for row in rows) / count,
        mrr=sum(row.mrr for row in rows) / count,
        ndcg_at_k=sum(row.ndcg_at_k for row in rows) / count,
    )
