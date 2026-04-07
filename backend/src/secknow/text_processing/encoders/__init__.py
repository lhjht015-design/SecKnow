from __future__ import annotations

"""编码层入口。"""

from .factory import build_embedder
from .sbert import DeterministicFakeEmbedder, SBERTEmbedder

__all__ = ["build_embedder", "SBERTEmbedder", "DeterministicFakeEmbedder"]
