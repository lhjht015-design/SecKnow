from __future__ import annotations

"""文档级元数据构建。"""

import time
from pathlib import Path

from ..config import SUPPORTED_EXTS
from ..schemas.document import LoadedDocument
from .hashing import build_doc_id


def build_loaded_document(file_path: str | Path) -> LoadedDocument:
    """从文件路径构造文档元数据。"""
    path = Path(file_path).expanduser().resolve()
    stat = path.stat()
    extension = path.suffix.lower()

    return LoadedDocument(
        source_path=str(path),
        filename=path.name,
        extension=extension,
        file_type=SUPPORTED_EXTS.get(extension, "text"),
        size_bytes=int(stat.st_size),
        mtime=int(stat.st_mtime),
        doc_id=build_doc_id(path),
    )


def build_loaded_document_for_text(source_path: str | Path, text: str) -> LoadedDocument:
    """为内存中的文本构造元数据，不要求路径在磁盘上真实存在。

    **用途**：`DocumentTextPipeline.process_text` 在不写盘的情况下仍需 `LoadedDocument`
    （doc_id、extension、file_type、size_bytes 等）以生成 `ChunkMetadata`。
    `mtime` 取当前时间；`size_bytes` 为 UTF-8 字节长度。
    """
    path = Path(source_path)
    extension = path.suffix.lower() or ".txt"
    file_type = SUPPORTED_EXTS.get(extension, "text")
    data = text.encode("utf-8")
    filename = path.name if path.name else "document.txt"

    return LoadedDocument(
        source_path=str(path),
        filename=filename,
        extension=extension,
        file_type=file_type,
        size_bytes=len(data),
        mtime=int(time.time()),
        doc_id=build_doc_id(path),
    )
