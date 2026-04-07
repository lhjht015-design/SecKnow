from __future__ import annotations

"""精确去重实现。"""

from ..metadata.hashing import build_content_hash


def dedup_exact(chunks: list[str]) -> list[str]:
    """按内容哈希去重，保留首次出现的 chunk。"""
    seen: set[str] = set()
    result: list[str] = []

    for chunk in chunks:
        normalized = chunk.strip()
        if not normalized:
            continue
        digest = build_content_hash(normalized)
        if digest in seen:
            continue
        seen.add(digest)
        result.append(normalized)

    return result
