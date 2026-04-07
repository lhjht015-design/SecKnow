from __future__ import annotations

"""分块路由。

**改动**：`_SUPPORTED` 在 `fixed_window` / `hybrid` 基础上增加 `paragraph`、`line`、`semantic`。
- `paragraph` / `line`：见同目录 `paragraph.py`、`line.py`。
- `semantic`：**LangChain SemanticChunker**（见 `semantic.py`），依赖真实嵌入，且会加载 LangChain 依赖树。
"""

from ..config import (
    DEFAULT_CHUNK_MAX_TOKENS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_STRATEGY,
)
from ..exceptions import ChunkingError
from .fixed_window import chunk_fixed_window, chunk_hybrid
from .line import chunk_by_line
from .paragraph import chunk_by_paragraph
from .semantic import chunk_semantic


_SUPPORTED = {"fixed_window", "hybrid", "paragraph", "line", "semantic"}


def chunk_text(
    text: str,
    strategy: str = DEFAULT_CHUNK_STRATEGY,
    max_tokens: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """根据策略执行分块。"""
    strategy = (strategy or DEFAULT_CHUNK_STRATEGY).strip().lower()
    max_tokens = max_tokens or DEFAULT_CHUNK_MAX_TOKENS
    overlap = DEFAULT_CHUNK_OVERLAP if overlap is None else overlap

    if strategy not in _SUPPORTED:
        raise ChunkingError(f"暂不支持的分块策略: {strategy}")

    if strategy == "fixed_window":
        return chunk_fixed_window(text=text, max_tokens=max_tokens, overlap=overlap)
    if strategy == "paragraph":
        return chunk_by_paragraph(text=text, max_tokens=max_tokens, overlap=overlap)
    if strategy == "line":
        return chunk_by_line(text=text, max_tokens=max_tokens, overlap=overlap)
    if strategy == "semantic":
        return chunk_semantic(text=text, max_tokens=max_tokens, overlap=overlap)
    return chunk_hybrid(text=text, max_tokens=max_tokens, overlap=overlap)
