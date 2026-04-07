from __future__ import annotations

"""加载路由：按扩展名选择具体加载器。"""

from pathlib import Path
from typing import Callable

from ..config import SUPPORTED_EXTS
from ..exceptions import DocumentLoadError, UnsupportedFileTypeError
from .code_loader import load_code
from .docx_loader import load_docx
from .markdown_loader import load_markdown
from .pdf_loader import load_pdf
from .text_loader import load_text

Loader = Callable[[str | Path], str]

_LOADER_MAP: dict[str, Loader] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".md": load_markdown,
    ".markdown": load_markdown,
}


def resolve_loader(file_path: str | Path) -> Loader:
    """根据扩展名解析对应加载函数。"""
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext in _LOADER_MAP:
        return _LOADER_MAP[ext]
    if ext in SUPPORTED_EXTS:
        if SUPPORTED_EXTS[ext] == "code":
            return load_code
        return load_text
    raise UnsupportedFileTypeError(f"不支持的文件类型: {ext or '<无扩展名>'}")


def load_document_text(file_path: str | Path) -> str:
    """读取文件并返回原始文本。"""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise DocumentLoadError(f"文件不存在或不可读: {path}")

    loader = resolve_loader(path)
    text = loader(path)
    return text
