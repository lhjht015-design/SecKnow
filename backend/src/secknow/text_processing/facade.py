from __future__ import annotations

"""4.1 对外门面：提供函数式接口和面向对象接口。

**近期 API 变动**：`load_document` 增加可选关键字参数 `normalize`（默认 False 保持旧行为）。
为 True 时走与 `run_pipeline` 相同的 `clean_document_text`（basic + 按扩展名类型清洗）。
"""

from pathlib import Path
from typing import Iterable

from .chunkers.router import chunk_text as _chunk_text
from .config import DEFAULT_CHUNK_STRATEGY, DEFAULT_DEDUP_STRATEGY
from .dedupers.router import dedup_chunks as _dedup_chunks
from .encoders.factory import build_embedder
from .loaders.router import load_document_text
from .pipeline.run_pipeline import DocumentTextPipeline, run_pipeline


def load_document(file: str | Path, *, normalize: bool = False) -> str:
    """读取文件并返回文本；normalize=True 时按扩展名做与流水线一致的基础 + 类型清洗。

    `normalize=False`：与历史行为一致，仅 loader 抽取原始文本（可能含冗余空白等）。
    """
    text = load_document_text(file)
    if not normalize:
        return text
    # 延迟导入：多数调用方只读原文时不必加载 cleaners 链。
    from .cleaners.document_clean import clean_document_text
    from .config import SUPPORTED_EXTS

    ext = Path(file).suffix.lower() or ".txt"
    file_type = SUPPORTED_EXTS.get(ext, "text")
    return clean_document_text(text, file_type)


def chunk_text(
    text: str,
    strategy: str | None = None,
    max_tokens: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """按策略对文本分块。"""
    return _chunk_text(
        text=text,
        strategy=strategy or DEFAULT_CHUNK_STRATEGY,
        max_tokens=max_tokens,
        overlap=overlap,
    )


def dedup(chunks: Iterable[str], strategy: str | None = None) -> list[str]:
    """按策略执行 chunk 去重。"""
    return _dedup_chunks(chunks=list(chunks), strategy=strategy or DEFAULT_DEDUP_STRATEGY)


def encode_chunks(
    chunks: Iterable[str],
    encoder=None,
    mode: str | None = None,
    model_name: str | None = None,
    dim: int | None = None,
) -> list[list[float]]:
    """对 chunk 列表做向量编码。"""
    embedder = encoder or build_embedder(mode=mode, model_name=model_name, dim=dim)
    return embedder.encode(list(chunks))


__all__ = [
    "DocumentTextPipeline",
    "load_document",
    "chunk_text",
    "dedup",
    "encode_chunks",
    "run_pipeline",
]
