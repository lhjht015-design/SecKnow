from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from secknow.stores.milvus import VectorHit


class LocalDenseStore:
    """本机文件后端的 dense 向量检索，用来替代 Milvus 的 dense 通道。

    为什么需要它
        原设计把 dense/sparse 向量放在 Milvus 里。Milvus 需要 Docker，而比赛/离线环境
        常常没有 Docker；一旦 Milvus 不可用，`RetrievalService` 会直接丢掉整个语义检索
        通道，只剩 FTS5 + TF-IDF 的词法召回。向量其实已经算出来了，白白浪费。

    它做什么
        从 `data/dense/<version>.npz` 载入全部块的 dense 向量，用矩阵乘法做精确余弦检索。
        12k–20k 个 1024 维向量的一次全量打分在 numpy 上是毫秒级，
        因此不需要 ANN 索引——精确检索反而更准。

    与 MilvusStore 的关系
        `hybrid_search()` 的返回结构与 `MilvusStore.hybrid_search()` 完全一致，
        因此可以在 `RetrievalService` 里直接替换，上层无需改动。
        sparse 通道返回空：词法召回已由 FTS5 与 TF-IDF 承担，本机无需再存一份稀疏向量。

    权重与一致性
        npz 由 `versions/v002_bulk_ingest.src` 在构建 TF-IDF 时一并写出，
        文件名就是 tfidf_version，因此向量与词法索引天然同版本、不会错配。
    """

    def __init__(
        self,
        directory: Path,
        version_provider: Callable[[], str | None],
    ) -> None:
        self.directory = directory
        self.version_provider = version_provider
        self._loaded_version: str | None = None
        self._chunk_ids: list[str] = []
        self._matrix: np.ndarray | None = None
        self._failed_version: str | None = None

    # -- 生命周期 ------------------------------------------------------------------

    def _path_for(self, version: str) -> Path:
        return self.directory / f"{version}.npz"

    def _ensure_loaded(self) -> bool:
        version = self.version_provider()
        if not version:
            return False
        if version == self._loaded_version and self._matrix is not None:
            return True
        # 同一个版本加载失败过就不再重试，避免每次检索都读一遍坏文件
        if version == self._failed_version:
            return False
        path = self._path_for(version)
        if not path.is_file():
            self._failed_version = version
            return False
        try:
            with np.load(path, allow_pickle=False) as payload:
                chunk_ids = [str(item) for item in payload["chunk_ids"]]
                matrix = np.asarray(payload["matrix"], dtype=np.float32)
        except (OSError, ValueError, KeyError):
            self._failed_version = version
            return False
        if matrix.ndim != 2 or matrix.shape[0] != len(chunk_ids):
            self._failed_version = version
            return False
        # 预先做 L2 归一化，之后点积即余弦相似度
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms
        self._chunk_ids = chunk_ids
        self._loaded_version = version
        self._failed_version = None
        return True

    def healthy(self) -> bool:
        return self._ensure_loaded()

    def close(self) -> None:
        self._matrix = None
        self._chunk_ids = []
        self._loaded_version = None

    # -- 检索 ----------------------------------------------------------------------

    def search(self, dense: Sequence[float], top_k: int = 40) -> list[VectorHit]:
        if not self._ensure_loaded() or self._matrix is None:
            return []
        query = np.asarray(dense, dtype=np.float32).ravel()
        if query.shape[0] != self._matrix.shape[1]:
            return []
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        scores = self._matrix @ (query / norm)
        limit = min(top_k, scores.shape[0])
        if limit <= 0:
            return []
        top = np.argpartition(-scores, limit - 1)[:limit]
        ordered = top[np.argsort(-scores[top], kind="stable")]
        return [VectorHit(self._chunk_ids[index], float(scores[index])) for index in ordered]

    def hybrid_search(
        self,
        dense: Sequence[float],
        sparse: dict[int, float],  # noqa: ARG002 - 保持与 MilvusStore 相同的签名
        top_k: int = 40,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,  # noqa: ARG002
    ) -> dict[str, list[VectorHit]]:
        """与 MilvusStore.hybrid_search 同构。sparse 通道留空（由 FTS/TF-IDF 承担）。"""
        hits = self.search(dense, top_k)
        # hybrid 通道在本机实现里就等于 dense：没有第二个向量通道需要融合
        return {"hybrid": hits, "dense": hits, "sparse": []}
