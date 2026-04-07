from __future__ import annotations

"""分块器抽象约束（占位）。"""

from typing import Protocol


class Chunker(Protocol):
    """统一分块协议。"""

    def __call__(self, text: str) -> list[str]:
        """执行分块。"""
        ...
