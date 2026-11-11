from __future__ import annotations

import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from secknow.domain.ids import normalize_text
from secknow.indexing.tfidf import security_tokens
from secknow.retrieval.reranker import RankedChunk


@dataclass(frozen=True)
class EvidenceCluster:
    cluster_id: int
    label: str
    score: float
    chunks: tuple[RankedChunk, ...]


def deduplicate(
    ranked: Sequence[RankedChunk],
    near_duplicate_threshold: float = 0.94,
) -> tuple[RankedChunk, ...]:
    if not 0 <= near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be between 0 and 1")
    ordered = [
        item
        for _, item in sorted(
            enumerate(ranked),
            key=lambda pair: (-pair[1].score, pair[0]),
        )
    ]
    seen_hashes: set[str] = set()
    seen_texts: set[str] = set()
    exact: list[RankedChunk] = []
    for item in ordered:
        normalized = "".join(
            character
            for character in normalize_text(item.chunk.text).casefold()
            if not unicodedata.category(character).startswith(("P", "Z"))
        )
        if item.chunk.content_sha256 in seen_hashes or normalized in seen_texts:
            continue
        seen_hashes.add(item.chunk.content_sha256)
        seen_texts.add(normalized)
        exact.append(item)
    if len(exact) < 2:
        return tuple(exact)

    try:
        matrix = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            norm="l2",
        ).fit_transform(item.chunk.text for item in exact)
    except ValueError:
        return tuple(exact)
    similarities = (matrix @ matrix.T).toarray()
    kept: list[int] = []
    for index in range(len(exact)):
        if any(similarities[index, previous] >= near_duplicate_threshold for previous in kept):
            continue
        kept.append(index)
    return tuple(exact[index] for index in kept)


def diversify(
    ranked: Sequence[RankedChunk],
    *,
    relevance_weight: float = 0.72,
    limit: int | None = None,
) -> tuple[RankedChunk, ...]:
    if not 0 <= relevance_weight <= 1:
        raise ValueError("relevance_weight must be between 0 and 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if len(ranked) < 2:
        return tuple(ranked[:limit])

    similarities = _text_similarities([item.chunk.text for item in ranked])
    scores = np.asarray([item.score for item in ranked], dtype=np.float64)
    highest = max(float(scores.max()), np.finfo(np.float64).eps)
    relevance = np.clip(scores / highest, 0.0, 1.0)
    remaining = set(range(len(ranked)))
    selected: list[int] = []
    target = min(limit or len(ranked), len(ranked))
    while remaining and len(selected) < target:
        def mmr(index: int) -> tuple[float, float, int]:
            redundancy = max((similarities[index, other] for other in selected), default=0.0)
            value = relevance_weight * relevance[index] - (1 - relevance_weight) * redundancy
            return float(value), float(relevance[index]), -index

        chosen = max(remaining, key=mmr)
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(ranked[index] for index in selected)


def regroup(
    ranked: Sequence[RankedChunk],
    max_clusters: int = 6,
    *,
    graph_distances: Mapping[str, int] | None = None,
) -> tuple[EvidenceCluster, ...]:
    if max_clusters < 1:
        raise ValueError("max_clusters must be positive")
    unique = diversify(deduplicate(ranked))
    if not unique:
        return ()
    if len(unique) == 1:
        return (EvidenceCluster(0, _label([unique[0].chunk.text]), unique[0].score, unique),)

    matrix = _cluster_matrix([item.chunk.text for item in unique])
    labels = _cluster_labels(matrix, max_clusters)
    grouped: dict[int, list[RankedChunk]] = defaultdict(list)
    for label, item in zip(labels, unique, strict=True):
        grouped[int(label)].append(item)

    diversity_rank = {item.chunk.chunk_id: index for index, item in enumerate(unique)}
    source_priority: dict[str, int] = {}
    for index, item in enumerate(unique):
        source_priority.setdefault(item.chunk.source_id, index)
    distances = graph_distances or {}
    clusters: list[EvidenceCluster] = []
    for cluster_id, items in grouped.items():
        items.sort(
            key=lambda item: (
                source_priority[item.chunk.source_id],
                max(0, distances.get(item.chunk.chunk_id, 0)),
                item.chunk.citation.page_start or 2**31,
                item.chunk.citation.line_start or 2**31,
                item.chunk.order,
                tuple(part.casefold() for part in item.chunk.section_path),
                diversity_rank[item.chunk.chunk_id],
            )
        )
        clusters.append(
            EvidenceCluster(
                cluster_id=cluster_id,
                label=_label([item.chunk.text for item in items]),
                score=max(item.score for item in items),
                chunks=tuple(items),
            )
        )
    return tuple(
        sorted(
            clusters,
            key=lambda cluster: (
                -cluster.score,
                min(diversity_rank[item.chunk.chunk_id] for item in cluster.chunks),
            ),
        )
    )


def _text_similarities(texts: Sequence[str]) -> np.ndarray:
    try:
        matrix = TfidfVectorizer(
            tokenizer=security_tokens,
            token_pattern=None,
            lowercase=False,
            sublinear_tf=True,
            norm="l2",
        ).fit_transform(texts)
    except ValueError:
        return np.eye(len(texts), dtype=np.float64)
    return cast(np.ndarray, (matrix @ matrix.T).toarray())


def _cluster_matrix(texts: Sequence[str]) -> np.ndarray:
    try:
        matrix = TfidfVectorizer(
            tokenizer=security_tokens,
            token_pattern=None,
            lowercase=False,
            max_features=2500,
        ).fit_transform(texts).toarray()
    except ValueError:
        return np.eye(len(texts), dtype=np.float64)
    zero_rows = np.flatnonzero(np.linalg.norm(matrix, axis=1) == 0)
    if not len(zero_rows):
        return cast(np.ndarray, matrix)
    fallback = np.zeros((len(texts), len(zero_rows)), dtype=matrix.dtype)
    fallback[zero_rows, np.arange(len(zero_rows))] = 1
    return np.hstack((matrix, fallback))


def _cluster_labels(matrix: np.ndarray, max_clusters: int) -> np.ndarray:
    labels = cast(
        np.ndarray,
        AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.68,
            metric="cosine",
            linkage="average",
        ).fit_predict(matrix),
    )
    if len(set(int(label) for label in labels)) <= max_clusters:
        return labels
    return cast(
        np.ndarray,
        AgglomerativeClustering(
            n_clusters=max_clusters,
            metric="cosine",
            linkage="average",
        ).fit_predict(matrix),
    )


def _label(texts: Sequence[str]) -> str:
    try:
        vectorizer = TfidfVectorizer(
            tokenizer=security_tokens,
            token_pattern=None,
            lowercase=False,
            max_features=500,
        )
        matrix = vectorizer.fit_transform(texts)
        weights = np.asarray(matrix.mean(axis=0)).ravel()
        names = vectorizer.get_feature_names_out()
        best = names[np.argsort(-weights)[:3]]
        return " · ".join(str(value) for value in best if value) or "相关证据"
    except ValueError:
        return "相关证据"
