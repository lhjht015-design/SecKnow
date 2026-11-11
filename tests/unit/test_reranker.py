from __future__ import annotations

from typing import cast

from secknow.domain.models import Chunk, Citation
from secknow.models.bge import BgeReranker
from secknow.retrieval.reranker import CandidateReranker


class FakeReranker:
    available = True

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        self.calls += 1
        return [float(index) for index, _ in enumerate(passages)]


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_id="document:test",
        source_version="version:test",
        text=text,
        content_sha256="0" * 64,
        element_ids=(f"element:{chunk_id}",),
        order=0,
        token_count=1,
        citation=Citation(
            source_id="document:test",
            source_version="version:test",
            title="test",
        ),
    )


def test_candidate_reranker_reuses_exact_candidate_set() -> None:
    model = FakeReranker()
    reranker = CandidateReranker(cast(BgeReranker, model), limit=2)
    candidates = [(_chunk("chunk:1", "first"), 0.5), (_chunk("chunk:2", "second"), 0.4)]

    first = reranker.rerank("query", candidates)
    second = reranker.rerank("query", candidates)

    assert first == second
    assert model.calls == 1


def test_candidate_reranker_invalidates_when_original_score_changes() -> None:
    model = FakeReranker()
    reranker = CandidateReranker(cast(BgeReranker, model), limit=1)
    chunk = _chunk("chunk:1", "first")

    reranker.rerank("query", [(chunk, 0.5)])
    reranker.rerank("query", [(chunk, 0.6)])

    assert model.calls == 2
