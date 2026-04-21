from __future__ import annotations

from .base import CodeScanner


class TextFallbackScanner(CodeScanner):
    """Simple line/block splitter used before AST-based scanning is implemented."""

    def __init__(self, language: str) -> None:
        self.language = language

    def extract_units(self, code_snippet: str, lang: str) -> list[str]:
        blocks = [part.strip() for part in code_snippet.split("\n\n")]
        return [block for block in blocks if block]
