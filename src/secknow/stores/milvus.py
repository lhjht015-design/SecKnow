from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from secknow.domain.models import Chunk


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    score: float


class MilvusStore:
    dense_field = "dense_vector"
    sparse_field = "sparse_vector"

    def __init__(
        self,
        uri: str,
        token: str | None = None,
        collection: str = "secknow_chunks",
        dimension: int = 1024,
    ) -> None:
        self.uri = uri
        self.token = token
        self.collection = collection
        self.dimension = dimension
        self._client: Any = None

    def connect(self) -> None:
        if self._client is not None:
            return
        try:
            from pymilvus import MilvusClient
        except ImportError as error:
            raise RuntimeError("Install SecKnow with the 'semantic' extra for Milvus") from error
        kwargs: dict[str, Any] = {"uri": self.uri}
        if self.token:
            kwargs["token"] = self.token
        self._client = MilvusClient(**kwargs)

    def healthy(self) -> bool:
        try:
            self.connect()
            self._client.list_collections()
            return True
        except Exception:
            return False

    def ensure_collection(self) -> None:
        self.connect()
        if self._client.has_collection(self.collection):
            return
        from pymilvus import DataType, MilvusClient

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("source_id", DataType.VARCHAR, max_length=128)
        schema.add_field("source_version", DataType.VARCHAR, max_length=128)
        schema.add_field("title", DataType.VARCHAR, max_length=1024)
        schema.add_field("section", DataType.VARCHAR, max_length=2048)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field(self.dense_field, DataType.FLOAT_VECTOR, dim=self.dimension)
        schema.add_field(self.sparse_field, DataType.SPARSE_FLOAT_VECTOR)
        index = self._client.prepare_index_params()
        index.add_index(
            field_name=self.dense_field,
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 24, "efConstruction": 240},
        )
        index.add_index(
            field_name=self.sparse_field,
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )
        self._client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index,
        )

    def upsert(
        self,
        chunks: Sequence[Chunk],
        dense: np.ndarray,
        sparse: Sequence[dict[int, float]],
    ) -> None:
        if len(chunks) != len(dense) or len(chunks) != len(sparse):
            raise ValueError("Chunks, dense vectors, and sparse vectors must have equal lengths")
        self.ensure_collection()
        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "source_version": chunk.source_version,
                "title": chunk.citation.title[:1024],
                "section": " / ".join(chunk.section_path)[:2048],
                "text": chunk.text[:65535],
                self.dense_field: dense[index].astype(np.float32).tolist(),
                self.sparse_field: sparse[index],
            }
            for index, chunk in enumerate(chunks)
        ]
        self._client.upsert(collection_name=self.collection, data=rows)

    def delete_source(self, source_id: str) -> None:
        self.connect()
        if not self._client.has_collection(self.collection):
            return
        escaped = source_id.replace('"', '\\"')
        self._client.delete(
            collection_name=self.collection,
            filter=f'source_id == "{escaped}"',
        )

    def delete_source_versions_except(self, source_id: str, version_id: str) -> None:
        self.connect()
        if not self._client.has_collection(self.collection):
            return
        escaped_source = source_id.replace('"', '\\"')
        escaped_version = version_id.replace('"', '\\"')
        self._client.delete(
            collection_name=self.collection,
            filter=(
                f'source_id == "{escaped_source}" and '
                f'source_version != "{escaped_version}"'
            ),
        )

    def hybrid_search(
        self,
        dense: Sequence[float],
        sparse: dict[int, float],
        top_k: int = 40,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> dict[str, list[VectorHit]]:
        self.connect()
        if not self._client.has_collection(self.collection):
            return {"hybrid": [], "dense": [], "sparse": []}
        if dense_weight < 0 or sparse_weight < 0 or dense_weight + sparse_weight == 0:
            raise ValueError("Hybrid search weights must be non-negative and not both zero")
        from pymilvus import AnnSearchRequest, WeightedRanker

        requests = [
            AnnSearchRequest(
                data=[list(dense)],
                anns_field=self.dense_field,
                param={"metric_type": "COSINE", "params": {"ef": max(64, top_k * 2)}},
                limit=top_k,
            ),
            AnnSearchRequest(
                data=[sparse],
                anns_field=self.sparse_field,
                param={"metric_type": "IP", "params": {}},
                limit=top_k,
            ),
        ]
        scale = max(dense_weight, sparse_weight)
        hybrid = self._client.hybrid_search(
            collection_name=self.collection,
            reqs=requests,
            ranker=WeightedRanker(dense_weight / scale, sparse_weight / scale),
            limit=top_k,
            output_fields=["chunk_id"],
        )
        output = {
            "hybrid": self._hits(hybrid),
            "dense": [],
            "sparse": [],
        }
        for channel, request in zip(("dense", "sparse"), requests, strict=True):
            result = self._client.search(
                collection_name=self.collection,
                data=request.data,
                anns_field=request.anns_field,
                search_params=request.param,
                limit=request.limit,
                output_fields=["chunk_id"],
            )
            output[channel] = self._hits(result)
        return output

    @staticmethod
    def _hits(result: Sequence[Sequence[dict[str, Any]]]) -> list[VectorHit]:
        if not result:
            return []
        return [
            VectorHit(str(hit["entity"]["chunk_id"]), float(hit["distance"])) for hit in result[0]
        ]
