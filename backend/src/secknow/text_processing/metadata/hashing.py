from __future__ import annotations

"""哈希相关工具。"""

from hashlib import sha1
from pathlib import Path


def build_doc_id(source_path: str | Path) -> str:
    """根据规范化路径生成稳定 doc_id。"""
    normalized = str(Path(source_path).expanduser().resolve())
    return sha1(normalized.encode("utf-8")).hexdigest()


def build_content_hash(text: str) -> str:
    """根据 chunk 文本生成稳定 content_hash。"""
    return sha1(text.strip().encode("utf-8")).hexdigest()
