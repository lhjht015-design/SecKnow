from __future__ import annotations

"""按行聚合分块：在不超过窗口的前提下合并连续行。

**改动**：由占位实现改为可用策略；适合日志、代码等多行短句场景。
单行超过 `max_tokens` 时对该行单独走 `chunk_fixed_window`。
"""

from ..config import DEFAULT_CHUNK_MAX_TOKENS, DEFAULT_CHUNK_OVERLAP
from .fixed_window import _to_tokens, chunk_fixed_window


def chunk_by_line(
    text: str,
    max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """每行一条逻辑记录，按 token 预算合并为多行块。"""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return []

    chunks: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if buffer:
            chunks.append("\n".join(buffer))
            buffer = []

    for line in lines:
        line_tokens = _to_tokens(line)
        if line_tokens.length > max_tokens:
            flush()
            chunks.extend(
                chunk_fixed_window(line, max_tokens=max_tokens, overlap=overlap)
            )
            continue

        candidate = "\n".join(buffer + [line]) if buffer else line
        if _to_tokens(candidate).length <= max_tokens:
            buffer.append(line)
        else:
            flush()
            buffer = [line]

    flush()
    return [c for c in chunks if c]
