from __future__ import annotations

from abc import ABC, abstractmethod


class CodeScanner(ABC):
    @abstractmethod
    def extract_units(self, code_snippet: str, lang: str) -> list[str]:
        """从代码中提取扫描单元，供后续检索和评分使用。"""
