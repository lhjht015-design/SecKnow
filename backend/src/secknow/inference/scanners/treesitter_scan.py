from __future__ import annotations

from .base import CodeScanner


class TreeSitterScanner(CodeScanner):
    """Reserved AST scanner entrypoint for a later implementation stage."""

    def extract_units(self, code_snippet: str, lang: str) -> list[str]:
        raise NotImplementedError("Tree-sitter scanning is not implemented yet.")
