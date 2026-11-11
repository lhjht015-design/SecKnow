from __future__ import annotations

import time

from secknow.config import Settings
from secknow.domain.models import SearchHit, SearchResponse
from secknow.graph.builder import KnowledgeGraph
from secknow.indexing.tfidf import TfidfIndex
from secknow.models.bge import BgeM3Encoder, BgeReranker
from secknow.retrieval.hybrid import query_channel_weights, weighted_rrf
from secknow.retrieval.reranker import CandidateReranker
from secknow.stores.dense import LocalDenseStore
from secknow.stores.metadata import MetadataStore
from secknow.stores.milvus import MilvusStore


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        store: MetadataStore | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or MetadataStore(settings.database_path)
        self.store.initialize()
        self.encoder = BgeM3Encoder(
            settings.bge_model_path,
            settings.device,
            settings.embedding_batch_size,
            model_revision=settings.bge_model_revision,
        )
        self.reranker = CandidateReranker(
            BgeReranker(settings.reranker_model_path, settings.device)
            if settings.reranker_enabled
            else None,
            limit=settings.reranker_top_k,
        )
        self.milvus = MilvusStore(
            settings.milvus_uri,
            settings.milvus_token.get_secret_value() if settings.milvus_token else None,
        )
        # Milvus 不可用时的本机向量后端，版本跟随当前活跃的 TF-IDF 版本
        self.dense = LocalDenseStore(
            settings.data_root / "dense",
            lambda: self.store.get_setting("active_tfidf_version"),
        )
        self._tfidf_cache: tuple[str, TfidfIndex] | None = None
        self._graph_cache: tuple[str, KnowledgeGraph] | None = None

    def search(self, query: str, top_k: int | None = None) -> SearchResponse:
        started = time.perf_counter()
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")
        limit = top_k or self.settings.result_top_k
        warnings: list[str] = []
        rankings: dict[str, list[tuple[str, float]]] = {}
        vector_channels: dict[str, dict[str, float]] = {}
        weights = query_channel_weights(query)

        rankings["bm25"] = self.store.search_fts(query, self.settings.retrieval_top_k)
        tfidf = self._load_tfidf()
        if tfidf is not None:
            rankings["tfidf"] = [
                (hit.chunk_id, hit.score)
                for hit in tfidf.search(query, self.settings.retrieval_top_k)
            ]
        else:
            warnings.append("TF-IDF 索引尚未构建")

        if self.settings.semantic_enabled:
            if not self.encoder.available:
                warnings.append("BGE-M3 未安装，已降级到词法检索")
            else:
                # 向量通道优先用 Milvus；没有 Docker 时退回本机 npz 后端。
                vector_store = None
                if self.milvus.healthy():
                    vector_store = self.milvus
                elif self.dense.healthy():
                    vector_store = self.dense
                    warnings.append("Milvus 不可用，向量通道已切换为本地 dense 索引")
                if vector_store is None:
                    warnings.append("向量索引不可用，已降级到本地词法检索")
                else:
                    try:
                        dense_vectors, sparse_vectors = self.encoder.encode([query])
                        vectors = vector_store.hybrid_search(
                            dense_vectors[0].tolist(),
                            sparse_vectors[0],
                            self.settings.retrieval_top_k,
                            weights["dense"],
                            weights["sparse"],
                        )
                        rankings["hybrid"] = [
                            (hit.chunk_id, hit.score) for hit in vectors["hybrid"]
                        ]
                        for channel in ("dense", "sparse"):
                            for hit in vectors[channel]:
                                vector_channels.setdefault(hit.chunk_id, {})[channel] = hit.score
                    except (RuntimeError, FileNotFoundError, MemoryError, OSError) as error:
                        warnings.append(f"语义检索降级：{type(error).__name__}")

        fused = weighted_rrf(rankings, {**weights, "hybrid": 1.0})
        chunk_map = self.store.get_chunks([candidate.item_id for candidate in fused])
        scored: dict[str, float] = {
            candidate.item_id: candidate.score
            for candidate in fused
            if candidate.item_id in chunk_map
        }
        paths: dict[str, tuple[str, ...]] = {}
        graph = self._load_graph()
        if graph is not None and scored:
            expanded = graph.expand(
                scored,
                self.settings.graph_max_hops,
                self.settings.graph_max_nodes,
            )
            for chunk_id, (graph_score, path) in expanded.items():
                if chunk_id not in chunk_map:
                    chunk_map.update(self.store.get_chunks([chunk_id]))
                if chunk_id in chunk_map:
                    scored[chunk_id] = max(scored.get(chunk_id, 0.0), graph_score)
                    paths[chunk_id] = path

        candidates = sorted(
            ((chunk_map[chunk_id], score) for chunk_id, score in scored.items()),
            key=lambda item: (-item[1], item[0].chunk_id),
        )
        reranked = self.reranker.rerank(query, candidates)
        channel_lookup = {candidate.item_id: candidate.channel_scores for candidate in fused}
        for chunk_id, scores in vector_channels.items():
            channel_lookup.setdefault(chunk_id, {}).update(scores)
        hits = tuple(
            SearchHit(
                chunk=item.chunk,
                score=item.score,
                rank=rank,
                channels=channel_lookup.get(item.chunk.chunk_id, {}),
                graph_paths=(paths[item.chunk.chunk_id],) if item.chunk.chunk_id in paths else (),
                explanation=self._explain(item.chunk.chunk_id, channel_lookup, paths),
            )
            for rank, item in enumerate(reranked[:limit], 1)
        )
        elapsed = (time.perf_counter() - started) * 1000
        return SearchResponse(
            query=query,
            mode=self.settings.mode.value,
            hits=hits,
            elapsed_ms=elapsed,
            ranking_score_calibrated=False,
            warnings=tuple(warnings),
        )

    def _load_tfidf(self) -> TfidfIndex | None:
        version = self.store.get_setting("active_tfidf_version")
        if not version:
            return None
        version_name = str(version)
        if self._tfidf_cache is not None and self._tfidf_cache[0] == version_name:
            return self._tfidf_cache[1]
        path = self.settings.data_root / "tfidf" / version_name
        if not path.exists():
            return None
        index = TfidfIndex.load(path)
        self._tfidf_cache = (version_name, index)
        return index

    def _load_graph(self) -> KnowledgeGraph | None:
        version = self.store.get_setting("active_graph_version")
        if not version:
            return None
        version_name = str(version)
        if self._graph_cache is not None and self._graph_cache[0] == version_name:
            return self._graph_cache[1]
        snapshot = self.settings.data_root / "graph" / f"{version_name}.json.gz"
        if not snapshot.exists():
            return None
        try:
            graph = KnowledgeGraph.load(snapshot)
            self._graph_cache = (version_name, graph)
            return graph
        except (OSError, ValueError):
            return None

    @staticmethod
    def _explain(
        chunk_id: str,
        channels: dict[str, dict[str, float]],
        paths: dict[str, tuple[str, ...]],
    ) -> str:
        scores = channels.get(chunk_id, {})
        metrics = {"fusion_relevance", "channel_coverage"}
        names = sorted(name for name in scores if name not in metrics)
        description = f"检索通道：{', '.join(names)}" if names else "图扩展证据"
        if "fusion_relevance" in scores:
            description += f"；融合相关度 {scores['fusion_relevance']:.3f}"
        if "channel_coverage" in scores:
            description += f"；通道覆盖度 {scores['channel_coverage']:.0%}"
        if chunk_id in paths and len(paths[chunk_id]) > 1:
            description += f"；图路径长度 {len(paths[chunk_id]) - 1}"
        return description
