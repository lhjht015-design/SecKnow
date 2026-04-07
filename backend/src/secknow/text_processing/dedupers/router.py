from __future__ import annotations

"""去重路由。

**改动**：由「仅支持 exact」扩展为 `exact | minhash`。
默认策略仍是 `exact`（`config.DEFAULT_DEDUP_STRATEGY`），不改变既有调用方行为。
"""

from ..config import DEFAULT_DEDUP_STRATEGY
from ..exceptions import TextProcessingError
from .exact import dedup_exact
from .minhash import dedup_minhash


def dedup_chunks(chunks: list[str], strategy: str = DEFAULT_DEDUP_STRATEGY) -> list[str]:
    """根据策略执行去重：exact（内容哈希）或 minhash（近似）。"""
    strategy = (strategy or DEFAULT_DEDUP_STRATEGY).strip().lower()
    if strategy == "exact":
        return dedup_exact(chunks)
    if strategy == "minhash":
        return dedup_minhash(chunks)
    raise TextProcessingError(f"暂不支持的去重策略: {strategy}")
