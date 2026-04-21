from __future__ import annotations

from secknow.text_processing.encoders.factory import build_embedder

from ..exceptions import QueryEncodingError


class QueryEncoder:
    """复用 4.1 编码基础设施完成查询向量化。"""

    def __init__(
        self,
        *,
        embedding_mode: str | None = None,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        self.embedder = build_embedder(
            mode=embedding_mode,
            model_name=embedding_model,
            dim=embedding_dim,
        )

    def encode(self, query: str) -> list[float]:
        text = query.strip()
        if not text:
            raise QueryEncodingError("query must not be empty")
        vectors = self.embedder.encode([text])
        if not vectors:
            raise QueryEncodingError("embedder returned no vectors")
        return vectors[0]
