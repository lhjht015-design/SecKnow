from __future__ import annotations

"""流水线结果结构定义。"""

from dataclasses import dataclass, field

from secknow.vector_store.models import ChunkRecord

from .chunk import EncodedChunk, TextChunk
from .document import LoadedDocument, RawDocument


@dataclass(slots=True)
class PipelineResult:
    """`process_file` 的完整结果。"""

    raw_document: RawDocument
    loaded_document: LoadedDocument
    chunks: list[TextChunk] = field(default_factory=list)
    deduped_chunks: list[TextChunk] = field(default_factory=list)
    encoded_chunks: list[EncodedChunk] = field(default_factory=list)
    records: list[ChunkRecord] = field(default_factory=list)

    @property
    def dropped_duplicates(self) -> int:
        """被去重移除的 chunk 数量。"""
        return max(0, len(self.chunks) - len(self.deduped_chunks))
