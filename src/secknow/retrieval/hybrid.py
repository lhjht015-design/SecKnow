from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import exp


@dataclass
class FusedCandidate:
    item_id: str
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)
    channel_scores: dict[str, float] = field(default_factory=dict)


def weighted_rrf(
    rankings: Mapping[str, Iterable[tuple[str, float]]],
    weights: Mapping[str, float] | None = None,
    rank_constant: int = 60,
) -> list[FusedCandidate]:
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    weights = weights or {name: 1.0 for name in rankings}
    candidates: dict[str, FusedCandidate] = {}
    active_weights: dict[str, float] = {}
    for channel, items in rankings.items():
        weight = weights.get(channel, 1.0)
        if weight < 0:
            raise ValueError("channel weights must be non-negative")
        materialized = list(items)
        if materialized and weight > 0:
            active_weights[channel] = weight
        for rank, (item_id, channel_score) in enumerate(materialized, 1):
            candidate = candidates.setdefault(item_id, FusedCandidate(item_id))
            candidate.score += weight / (rank_constant + rank)
            candidate.ranks[channel] = rank
            candidate.channel_scores[channel] = channel_score
    if not candidates:
        return []

    # RRF is the relevance signal. Channel coverage is a label-free recall proxy:
    # candidates independently found by more active retrievers receive a small boost.
    max_rrf = max(candidate.score for candidate in candidates.values()) or 1.0
    total_weight = sum(active_weights.values()) or 1.0
    for candidate in candidates.values():
        relevance = candidate.score / max_rrf
        coverage = sum(active_weights.get(name, 0.0) for name in candidate.ranks) / total_weight
        candidate.channel_scores["fusion_relevance"] = relevance
        candidate.channel_scores["channel_coverage"] = coverage
        candidate.score = 0.78 * relevance + 0.22 * coverage
    return sorted(candidates.values(), key=lambda item: (-item.score, item.item_id))


@dataclass(frozen=True)
class ScoreCalibrator:
    intercept: float
    coefficients: dict[str, float]
    version: str

    def predict(self, features: dict[str, float]) -> float:
        value = self.intercept + sum(
            coefficient * features.get(name, 0.0) for name, coefficient in self.coefficients.items()
        )
        return 1 / (1 + exp(-max(-40.0, min(40.0, value))))


def query_channel_weights(query: str) -> dict[str, float]:
    lowered = query.casefold()
    exact_markers = ("cve-", "cwe-", "capec-", "0x", "::", "()", "/", "rax", "rsp")
    exact = any(marker in lowered for marker in exact_markers)
    code_like = sum(character in "_./:+-#" for character in query) >= 2
    if exact or code_like:
        return {"bm25": 1.4, "tfidf": 1.35, "dense": 0.8, "sparse": 1.25, "graph": 0.9}
    return {"bm25": 1.0, "tfidf": 1.0, "dense": 1.25, "sparse": 1.1, "graph": 1.0}


def merge_channel_scores(candidates: Iterable[FusedCandidate]) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = defaultdict(dict)
    for candidate in candidates:
        merged[candidate.item_id].update(candidate.channel_scores)
        merged[candidate.item_id]["rrf"] = candidate.score
    return dict(merged)
