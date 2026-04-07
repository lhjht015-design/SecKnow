from __future__ import annotations

"""SBERT 与 fake embedding 编码器。"""

import hashlib
from math import sqrt

from ..config import DEFAULT_EMBEDDING_DIM, DEFAULT_EMBEDDING_MODEL
from ..exceptions import EncodingError

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except Exception:  # pragma: no cover - 依赖缺失时由运行时异常覆盖
    _SentenceTransformer = None


class DeterministicFakeEmbedder:
    """确定性伪向量编码器（用于联调与测试）。"""

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM):
        self.dim = dim

    def _encode_one(self, text: str) -> list[float]:
        """把单条文本编码为固定维度向量。"""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [(digest[i % len(digest)] / 255.0) for i in range(self.dim)]
        norm = sqrt(sum(v * v for v in raw))
        if norm == 0:
            return raw
        return [v / norm for v in raw]

    def encode(self, texts: list[str]) -> list[list[float]]:
        """批量编码文本。"""
        return [self._encode_one(text) for text in texts]


class SBERTEmbedder:
    """基于 sentence-transformers 的 SBERT 编码器。"""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        dim: int = DEFAULT_EMBEDDING_DIM,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ):
        if _SentenceTransformer is None:
            raise EncodingError(
                "未检测到 sentence-transformers，请先安装依赖或改用 EMBEDDING_MODE=fake"
            )

        self.model_name = model_name
        self.dim = dim
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

        try:
            self._model = _SentenceTransformer(model_name)
        except Exception as exc:  # pragma: no cover - 三方库异常类型不稳定
            raise EncodingError(f"加载 SBERT 模型失败: {model_name}") from exc

    def _align_dim(self, vector: list[float]) -> list[float]:
        """把向量对齐到目标维度，保证 Phase 1 与 4.3 的维度契约稳定。"""
        if len(vector) == self.dim:
            return vector
        if len(vector) > self.dim:
            return vector[: self.dim]
        return vector + [0.0] * (self.dim - len(vector))

    def encode(self, texts: list[str]) -> list[list[float]]:
        """批量编码文本为向量。"""
        if not texts:
            return []

        try:
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
                batch_size=self.batch_size,
            )
        except Exception as exc:  # pragma: no cover - 三方库异常类型不稳定
            raise EncodingError("SBERT 编码执行失败") from exc

        vectors: list[list[float]] = []
        for item in embeddings:
            vector = [float(x) for x in item]
            vectors.append(self._align_dim(vector))
        return vectors
