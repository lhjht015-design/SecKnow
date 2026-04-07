from __future__ import annotations

"""按段落分块：双换行分段，超长段退回固定窗口。

**改动**：由占位实现改为可用策略；经 `chunkers/router.chunk_text(..., strategy="paragraph")` 暴露。
若仅有单换行而无空行，会退化为按单行拆段（适配经 basic_clean 后空行被压缩的文本）。
"""

import re

from ..config import DEFAULT_CHUNK_MAX_TOKENS, DEFAULT_CHUNK_OVERLAP
from .fixed_window import _to_tokens, chunk_fixed_window


def chunk_by_paragraph(
    text: str,
    max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """按空行分段；单段过长则用固定窗口切分。"""
    compact = text.strip()
    if not compact:
        return []

    parts = [p.strip() for p in re.split(r"\n\s*\n+", compact) if p.strip()]
    if len(parts) == 1 and "\n" in parts[0]:
        parts = [p.strip() for p in parts[0].split("\n") if p.strip()]

    chunks: list[str] = []
    for paragraph in parts:
        token_seq = _to_tokens(paragraph)
        if token_seq.length > max_tokens:
            chunks.extend(
                chunk_fixed_window(paragraph, max_tokens=max_tokens, overlap=overlap)
            )
        else:
            chunks.append(paragraph)
    return [c for c in chunks if c]
