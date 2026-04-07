from __future__ import annotations

from pathlib import Path

from secknow.vector_store.services.sparse import MemoryBm25Index, SQLiteFtsSparseIndex
from secknow.vector_store.services.vector_service import VectorInfrastructureService
from secknow.vector_store.stores.faiss_sqlite_store import FaissSqliteVectorStore
from secknow.vector_store.stores.qdrant_store import QdrantVectorStore


def build_online_service(
    *,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    distance: str = "cosine",
) -> VectorInfrastructureService:
    dense_store = QdrantVectorStore(
        host=qdrant_host,
        port=qdrant_port,
        embedding_model=embedding_model,
        distance=distance,
    )
    sparse_index = MemoryBm25Index()
    return VectorInfrastructureService(dense_store=dense_store, sparse_index=sparse_index)


def build_offline_service(
    *,
    db_path: str | Path = "db/secknow.sqlite3",
    index_dir: str | Path = "db/faiss",
    embedding_dim: int = 384,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    distance: str = "cosine",
) -> VectorInfrastructureService:
    dense_store = FaissSqliteVectorStore(
        db_path=db_path,
        index_dir=index_dir,
        embedding_dim=embedding_dim,
        embedding_model=embedding_model,
        distance=distance,
    )
    sparse_index = SQLiteFtsSparseIndex(db_path=db_path)
    return VectorInfrastructureService(dense_store=dense_store, sparse_index=sparse_index)


def build_vector_service(
    *,
    mode: str = "online",
    **kwargs: object,
) -> VectorInfrastructureService:
    if mode == "online":
        return build_online_service(**kwargs)
    if mode == "offline":
        return build_offline_service(**kwargs)
    raise ValueError("mode must be either 'online' or 'offline'")
