from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from secknow.domain.models import Chunk
from secknow.models.bge import BgeReranker

_CAPTION = re.compile(r"!\[([^\]]*)\]\(")


def table_view(text: str) -> str:
    """表格块的重排视图：图注 + 单元格文本，剥掉 OCR 碎行和图片路径。

    表格块的全文常被 OCR 碎片与路径噪声占据，交叉编码器在截断窗口里
    读不到表格本身；这个视图把表格内容提到最前面。
    """
    captions = [match.group(1) for match in _CAPTION.finditer(text)]
    rows: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = stripped.strip("|").replace("|", " ").split()
        if not cells or all(set(cell) <= set("-") for cell in cells):
            continue
        rows.append(" ".join(cells))
    if not rows:
        return text
    head = "\n".join(caption.strip() for caption in captions if caption.strip())
    return f"{head}\n" + "\n".join(rows) if head else "\n".join(rows)


def _passages(chunk: Chunk) -> list[str]:
    texts = [chunk.text]
    if "table" in chunk.metadata.get("element_kinds", ()):
        view = table_view(chunk.text)
        if view != chunk.text:
            texts.append(view)
    return texts


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
        passage_sets = [_passages(chunk) for chunk, _ in limited]
        try:
            raw_scores = self.model.rerank(
                query, [passage for passages in passage_sets for passage in passages]
            )
        except (RuntimeError, FileNotFoundError, MemoryError):
            ranked = tuple(
                RankedChunk(chunk, score, score)
                for chunk, score in sorted(limited, key=lambda item: -item[1])
            )
            self._remember(cache_key, ranked)
            return ranked
        # 一个块有多个视图时取最高分：表格块只要全文或表格视图之一强相关即可，
        # 避免 OCR 碎行把表格本身的匹配压下去。
        best_raw: list[float] = []
        offset = 0
        for passages in passage_sets:
            best_raw.append(max(raw_scores[offset : offset + len(passages)]))
            offset += len(passages)
        normalized = 1 / (1 + np.exp(-np.asarray(best_raw, dtype=np.float32)))
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
