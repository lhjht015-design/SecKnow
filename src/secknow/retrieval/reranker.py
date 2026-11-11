from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from secknow.domain.models import Chunk
from secknow.models.bge import BgeReranker


@dataclass(frozen=True)
class RankedChunk:
    chunk: Chunk
    score: float
    original_score: float


class CandidateReranker:
    def __init__(
        self,
        model: BgeReranker | None = None,
        limit: int = 24,
        cache_size: int = 128,
    ) -> None:
        self.model = model
        self.limit = limit
        self.cache_size = max(0, cache_size)
        self._cache: OrderedDict[
            tuple[str, tuple[tuple[str, float], ...]], tuple[RankedChunk, ...]
        ] = OrderedDict()

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[Chunk, float]],
    ) -> tuple[RankedChunk, ...]:
        limited = list(candidates[: self.limit])
        if not limited:
            return ()
        cache_key = (
            query,
            tuple((chunk.chunk_id, round(score, 8)) for chunk, score in limited),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached
        if self.model is None or not self.model.available:
            ranked = tuple(
                RankedChunk(chunk, score, score)
                for chunk, score in sorted(limited, key=lambda item: -item[1])
            )
            self._remember(cache_key, ranked)
            return ranked
        try:
            raw_scores = self.model.rerank(query, [chunk.text for chunk, _ in limited])
        except (RuntimeError, FileNotFoundError, MemoryError):
            ranked = tuple(
                RankedChunk(chunk, score, score)
                for chunk, score in sorted(limited, key=lambda item: -item[1])
            )
            self._remember(cache_key, ranked)
            return ranked
        normalized = 1 / (1 + np.exp(-np.asarray(raw_scores, dtype=np.float32)))
        ranked_rows = [
            RankedChunk(chunk, float(0.72 * rerank + 0.28 * original), original)
            for (chunk, original), rerank in zip(limited, normalized, strict=True)
        ]
        result = tuple(sorted(ranked_rows, key=lambda item: (-item.score, item.chunk.chunk_id)))
        self._remember(cache_key, result)
        return result

    def _remember(
        self,
        key: tuple[str, tuple[tuple[str, float], ...]],
        value: tuple[RankedChunk, ...],
    ) -> None:
        if self.cache_size <= 0:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
