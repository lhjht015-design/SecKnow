from __future__ import annotations

"""元数据构建层入口。"""

from .chunk import build_chunk_metadata
from .document import build_loaded_document
from .hashing import build_content_hash, build_doc_id

__all__ = [
    "build_doc_id",
    "build_content_hash",
    "build_loaded_document",
    "build_chunk_metadata",
]
