from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from secknow.vector_store.models import (
    BaselineBundle,
    ChunkRecord,
    DeleteResult,
    SearchHit,
    UpsertResult,
    ZoneId,
)


FILTERABLE_FIELDS = frozenset(
    {
        "doc_id",
        "filename",
        "source_path",
        "extension",
        "file_type",
        "language",
        "record_type",
    }
)


def normalize_search_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    """Apply the shared 4.3 filtering contract across all backends."""
    normalized: dict[str, Any] = {}
    for key, value in (filters or {}).items():
        if key not in FILTERABLE_FIELDS:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[key] = value
    normalized.setdefault("record_type", "knowledge")
    return normalized


class VectorStore(ABC):
    @abstractmethod
    def ensure_zone(self, zone_id: ZoneId, dim: int) -> None:
        """确保分区对应的存储后端存在，且向量维度匹配。"""

    @abstractmethod
    def upsert(self, zone_id: ZoneId, records: list[ChunkRecord]) -> UpsertResult:
        """向分区插入或更新 chunk 向量与元数据。"""

    @abstractmethod
    def search(
        self,
        zone_id: ZoneId,
        query_vec: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """在分区内执行 ANN 检索。"""

    @abstractmethod
    def delete(self, zone_id: ZoneId, chunk_ids: list[str]) -> DeleteResult:
        """按外部 chunk_id 删除 chunk。"""

    @abstractmethod
    def delete_by_doc_id(self, zone_id: ZoneId, doc_id: str) -> DeleteResult:
        """按文档级 doc_id 删除该文档的全部 chunk。"""

    @abstractmethod
    def get_baseline(self, zone_id: ZoneId) -> BaselineBundle:
        """为安全模块加载基线向量。"""

    @abstractmethod
    def export_zone(self, zone_id: ZoneId, target_dir: str) -> dict[str, Any]:
        """将分区导出为离线包产物。"""
