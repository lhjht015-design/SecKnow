from __future__ import annotations

from typing import cast

from secknow.domain.models import Chunk, Citation
from secknow.models.bge import BgeReranker
from secknow.retrieval.reranker import CandidateReranker, table_view


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


def test_table_view_keeps_caption_and_cells_without_paths() -> None:
    text = (
        "垃圾 OCR 行\n\n"
        "![表5-1-2 常用指令](.secknow-media/p0412-01.png)\n\n"
        "| 操作码 | 指令示例 |\n| --- | --- |\n| mov | mov rax, rbx |"
    )

    view = table_view(text)

    assert "表5-1-2 常用指令" in view
    assert "mov rax, rbx" in view
    assert ".secknow-media" not in view
    assert "垃圾" not in view


class ViewAwareReranker:
    available = True

    def __init__(self) -> None:
        self.passages: list[str] = []

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        self.passages = passages
        # 表格视图已剥掉图片路径，全文视图还带着
        return [4.0 if ".secknow-media" not in passage else -6.0 for passage in passages]


def test_table_chunk_takes_best_of_full_text_and_table_view() -> None:
    model = ViewAwareReranker()
    reranker = CandidateReranker(cast(BgeReranker, model), limit=1)
    chunk = _chunk(
        "chunk:table",
        "垃圾 OCR 行\n\n"
        "![表5-1-2 常用指令](.secknow-media/p0412-01.png)\n\n"
        "| 操作码 | 指令示例 |\n| --- | --- |\n| mov | mov rax, rbx |",
    ).model_copy(update={"metadata": {"element_kinds": ["table"]}})

    result = reranker.rerank("x86 指令操作码", [(chunk, 0.0)])

    assert len(model.passages) == 2
    assert result[0].score > 0.7
