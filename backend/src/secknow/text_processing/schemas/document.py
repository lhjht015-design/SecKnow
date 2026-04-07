from __future__ import annotations

"""文档内部结构定义。"""

from dataclasses import dataclass


@dataclass(slots=True)
class RawDocument:
    """文件读取后的原始文本。"""

    source_path: str
    text: str


@dataclass(slots=True)
class LoadedDocument:
    """文档元数据。"""

    source_path: str
    filename: str
    extension: str
    file_type: str
    size_bytes: int
    mtime: int
    doc_id: str
