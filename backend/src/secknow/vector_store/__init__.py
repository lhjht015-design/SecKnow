"""SecKnow 4.3 向量存储模块包。"""

from secknow.vector_store.factory import (
    build_offline_service,
    build_online_service,
    build_vector_service,
)
from secknow.vector_store.models import (
    BaselineBundle,
    ChunkMetadata,
    ChunkRecord,
    DeleteResult,
    ExportManifest,
    SearchHit,
    SparseHit,
    UpsertResult,
    generate_chunk_id,
)
from secknow.vector_store.services.vector_service import VectorInfrastructureService

VectorService = VectorInfrastructureService

__all__ = [
    "BaselineBundle",
    "ChunkMetadata",
    "ChunkRecord",
    "DeleteResult",
    "ExportManifest",
    "SearchHit",
    "SparseHit",
    "UpsertResult",
    "VectorService",
    "build_online_service",
    "build_offline_service",
    "build_vector_service",
    "generate_chunk_id",
]
