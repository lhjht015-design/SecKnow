from __future__ import annotations

from .base import CodeScanner


class TextFallbackScanner(CodeScanner):
    """在 AST 扫描落地前使用的简单文本分块扫描器。"""

    def __init__(self, language: str) -> None:
        self.language = language

    def extract_units(self, code_snippet: str, lang: str) -> list[str]:
        blocks = [part.strip() for part in code_snippet.split("\n\n")]
        return [block for block in blocks if block]
