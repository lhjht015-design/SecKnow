from __future__ import annotations

from abc import ABC, abstractmethod


class CodeScanner(ABC):
    @abstractmethod
    def extract_units(self, code_snippet: str, lang: str) -> list[str]:
        """Extract scan units from code for later retrieval and scoring."""
