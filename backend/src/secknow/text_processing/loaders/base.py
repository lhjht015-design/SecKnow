from __future__ import annotations

"""加载器抽象约束（占位）。"""

from pathlib import Path
from typing import Protocol


class TextLoader(Protocol):
    """统一加载器协议：输入文件路径，输出原始文本。"""

    def __call__(self, file_path: str | Path) -> str:
        """执行文本加载。"""
        ...
