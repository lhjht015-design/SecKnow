from __future__ import annotations

"""分块单测。

semantic 相关用例通过 patch LangChain，避免 CI 下载句向量模型；真实语义分块行为以集成环境验证为准。
"""

from unittest.mock import MagicMock, patch

import pytest

from secknow.text_processing.chunkers.router import chunk_text
from secknow.text_processing.exceptions import ChunkingError


def test_fixed_window_chunking() -> None:
    """fixed_window 应按窗口切分文本。"""
    text = " ".join([f"token{i}" for i in range(20)])
    chunks = chunk_text(text, strategy="fixed_window", max_tokens=6, overlap=2)
    assert len(chunks) >= 3
    assert all(chunk for chunk in chunks)


def test_hybrid_chunking() -> None:
    """hybrid 应支持段落聚合与切分。"""
    text = "第一段内容\n第二段内容\n第三段内容"
    chunks = chunk_text(text, strategy="hybrid", max_tokens=10, overlap=2)
    assert len(chunks) >= 1
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_paragraph_chunking() -> None:
    """paragraph 应按空行分段。"""
    text = "段落一内容\n\n段落二内容"
    chunks = chunk_text(text, strategy="paragraph", max_tokens=50, overlap=2)
    assert len(chunks) >= 2


def test_line_chunking() -> None:
    """line 应合并相邻行。"""
    text = "line1\nline2\nline3"
    chunks = chunk_text(text, strategy="line", max_tokens=20, overlap=2)
    assert len(chunks) >= 1


def test_semantic_chunking_uses_langchain_semantic_chunker(monkeypatch) -> None:
    """semantic 应走 LangChain SemanticChunker（嵌入驱动语义断点）。"""
    monkeypatch.delenv("EMBEDDING_MODE", raising=False)

    fake_doc_a = MagicMock()
    fake_doc_a.page_content = "语义块甲"
    fake_doc_b = MagicMock()
    fake_doc_b.page_content = "语义块乙"

    class FakeChunker:
        def __init__(self, *args, **kwargs):
            pass

        def create_documents(self, texts: list[str]):
            assert texts == ["第一句。第二句。"]
            return [fake_doc_a, fake_doc_b]

    mock_embeddings = MagicMock()
    with (
        patch(
            "langchain_experimental.text_splitter.SemanticChunker", FakeChunker
        ),
        patch(
            "secknow.text_processing.chunkers.semantic._build_hf_embeddings",
            lambda *_a, **_k: mock_embeddings,
        ),
    ):
        chunks = chunk_text(
            "第一句。第二句。", strategy="semantic", max_tokens=80, overlap=2
        )
    assert chunks == ["语义块甲", "语义块乙"]


def test_semantic_chunking_rejects_fake_embedding(monkeypatch) -> None:
    """fake 向量模式下不能使用语义分块。"""
    monkeypatch.setenv("EMBEDDING_MODE", "fake")
    with pytest.raises(ChunkingError, match="semantic"):
        chunk_text("你好。世界。", strategy="semantic", max_tokens=20, overlap=2)
