from __future__ import annotations

"""4.1 内部数据结构。"""

from .chunk import EncodedChunk, TextChunk
from .document import LoadedDocument, RawDocument
from .pipeline_result import PipelineResult

__all__ = [
    "RawDocument",
    "LoadedDocument",
    "TextChunk",
    "EncodedChunk",
    "PipelineResult",
]
