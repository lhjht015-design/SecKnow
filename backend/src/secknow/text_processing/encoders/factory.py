from __future__ import annotations

"""编码器工厂。"""

from ..config import get_embedding_dim, get_embedding_mode, get_embedding_model
from ..exceptions import EncodingError
from .sbert import DeterministicFakeEmbedder, SBERTEmbedder


def build_embedder(
    mode: str | None = None,
    model_name: str | None = None,
    dim: int | None = None,
):
    """根据配置构建 embedding 编码器。"""
    resolved_mode = (mode or get_embedding_mode()).strip().lower()
    resolved_model_name = model_name or get_embedding_model()
    resolved_dim = dim or get_embedding_dim()

    if resolved_mode == "fake":
        return DeterministicFakeEmbedder(dim=resolved_dim)
    if resolved_mode in {"sbert", "real", "sentence_transformers"}:
        return SBERTEmbedder(model_name=resolved_model_name, dim=resolved_dim)

    raise EncodingError(f"不支持的 EMBEDDING_MODE: {resolved_mode}")
