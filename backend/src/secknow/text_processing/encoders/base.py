from __future__ import annotations

"""编码器抽象约束（占位）。"""

from typing import Protocol


class Embedder(Protocol):
    """统一编码器协议。"""

    def encode(self, texts: list[str]) -> list[list[float]]:
        """批量把文本编码为向量。"""
        ...
