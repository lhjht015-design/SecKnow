from __future__ import annotations

"""Markdown 加载器。"""

from pathlib import Path

from .text_loader import load_text


def load_markdown(file_path: str | Path) -> str:
    """读取 Markdown 原文。"""
    return load_text(file_path)
