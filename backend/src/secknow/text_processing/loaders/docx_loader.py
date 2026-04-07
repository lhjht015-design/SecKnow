from __future__ import annotations

"""DOCX 加载器。"""

from pathlib import Path

from docx import Document

from ..exceptions import DocumentLoadError


def load_docx(file_path: str | Path) -> str:
    """提取 DOCX 文本，按段落拼接。"""
    path = Path(file_path)
    try:
        doc = Document(path)
    except Exception as exc:  # pragma: no cover - 三方库抛错类型不稳定
        raise DocumentLoadError(f"读取 DOCX 失败: {path}") from exc

    paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text]
    return "\n".join(paragraphs)
