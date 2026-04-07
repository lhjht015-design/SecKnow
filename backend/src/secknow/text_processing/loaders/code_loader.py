from __future__ import annotations

"""代码文件加载器。"""

from pathlib import Path

from .text_loader import load_text


def load_code(file_path: str | Path) -> str:
    """读取代码文件原文。"""
    return load_text(file_path)
