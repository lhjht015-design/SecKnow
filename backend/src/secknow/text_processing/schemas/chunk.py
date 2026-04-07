from __future__ import annotations

"""chunk 内部结构定义。"""

from dataclasses import dataclass, field


@dataclass(slots=True)
class TextChunk:
    """文本分块。"""

    text: str
    chunk_index: int
    chunk_count: int = 0

    @property
    def char_len(self) -> int:
        """当前 chunk 字符数。"""
        return len(self.text)


@dataclass(slots=True)
class EncodedChunk:
    """带向量的 chunk。"""

    text_chunk: TextChunk
    vector: list[float] = field(default_factory=list)
