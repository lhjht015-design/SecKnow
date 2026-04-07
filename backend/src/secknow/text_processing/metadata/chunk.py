from __future__ import annotations

"""chunk 级元数据构建。"""

from secknow.vector_store.models import ChunkMetadata, RecordType, ZoneId

from ..schemas.document import LoadedDocument
from .hashing import build_content_hash


def build_chunk_metadata(
    *,
    document: LoadedDocument,
    zone_id: ZoneId,
    chunk_index: int,
    chunk_count: int,
    chunk_text: str,
    record_type: RecordType = "knowledge",
    language: str | None = None,
) -> ChunkMetadata:
    """构造单条 chunk 的标准元数据。"""
    return ChunkMetadata(
        doc_id=document.doc_id,
        zone_id=zone_id,
        filename=document.filename,
        source_path=document.source_path,
        extension=document.extension,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        char_len=len(chunk_text),
        content_hash=build_content_hash(chunk_text),
        mtime=document.mtime,
        size_bytes=document.size_bytes,
        file_type=document.file_type,
        language=language,
        record_type=record_type,
    )
