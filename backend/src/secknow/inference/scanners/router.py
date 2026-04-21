from __future__ import annotations

from .base import CodeScanner
from .text_fallback import TextFallbackScanner
from .treesitter_scan import TreeSitterScanner


def build_code_scanner(lang: str, *, prefer_ast: bool = False) -> CodeScanner:
    normalized = lang.strip().lower()
    if prefer_ast:
        return TreeSitterScanner()
    return TextFallbackScanner(language=normalized)
