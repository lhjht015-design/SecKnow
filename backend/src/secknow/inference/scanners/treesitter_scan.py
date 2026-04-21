from __future__ import annotations

from .base import CodeScanner


class TreeSitterScanner(CodeScanner):
    """预留给后续阶段的 AST 扫描入口。"""

    def extract_units(self, code_snippet: str, lang: str) -> list[str]:
        raise NotImplementedError("Tree-sitter scanning is not implemented yet.")
