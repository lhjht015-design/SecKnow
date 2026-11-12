from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np

from secknow.domain.ids import sha256_bytes, stable_id
from secknow.domain.models import Chunk, CodeSymbol, Entity, Relation
from secknow.graph.entities import EntityExtractor
from secknow.graph.keywords import KeywordScore, extract_keywords


class KnowledgeGraph:
    def __init__(
        self,
        extractor: EntityExtractor | None = None,
        *,
        keyword_top_k: int = 6,
        shared_relation_limit: int = 12,
    ) -> None:
        self.graph = nx.MultiDiGraph()
        self.extractor = extractor or EntityExtractor()
        self.keyword_top_k = keyword_top_k
        self.shared_relation_limit = shared_relation_limit

    def build(self, chunks: Sequence[Chunk], symbols: Sequence[CodeSymbol] = ()) -> None:
        self.graph.clear()
        entity_chunks: dict[str, set[str]] = defaultdict(set)
        entities: dict[str, Entity] = {}
        keyword_chunks: dict[str, set[str]] = defaultdict(set)
        keyword_scores: dict[tuple[str, str], KeywordScore] = {}
        element_chunks: dict[str, set[str]] = defaultdict(set)

        per_chunk_keywords = extract_keywords(
            [chunk.text for chunk in chunks],
            top_k=self.keyword_top_k,
        )
        for chunk, keywords in zip(chunks, per_chunk_keywords, strict=True):
            document_id = self._document_id(chunk)
            self.graph.add_node(
                document_id,
                kind="document",
                source_id=chunk.source_id,
                source_version=chunk.source_version,
                title=chunk.citation.title,
            )
            self.graph.add_node(
                chunk.chunk_id,
                kind="chunk",
                source_id=chunk.source_id,
                source_version=chunk.source_version,
                order=chunk.order,
                title=chunk.citation.title,
                page=chunk.citation.page_start,
            )
            self._add_hierarchy(document_id, chunk)
            if chunk.previous_chunk_id:
                self._bidirectional_edge(
                    chunk.previous_chunk_id,
                    chunk.chunk_id,
                    "NEXT",
                    "PREVIOUS",
                    weight=1.0,
                    confidence=1.0,
                )
            for element_id in chunk.element_ids:
                element_chunks[element_id].add(chunk.chunk_id)
            for mention in self.extractor.extract(chunk.text):
                entity = mention.entity
                entities[entity.entity_id] = entity
                entity_chunks[entity.entity_id].add(chunk.chunk_id)
                self.graph.add_node(
                    entity.entity_id,
                    kind="entity",
                    entity_kind=entity.kind,
                    name=entity.canonical_name,
                )
                self._bidirectional_edge(
                    chunk.chunk_id,
                    entity.entity_id,
                    "MENTIONS",
                    "MENTIONED_BY",
                    weight=1.0,
                    confidence=entity.confidence,
                )
            for keyword in keywords:
                keyword_id = stable_id("keyword", keyword.token)
                keyword_chunks[keyword_id].add(chunk.chunk_id)
                keyword_scores[(chunk.chunk_id, keyword_id)] = keyword
                self.graph.add_node(
                    keyword_id,
                    kind="keyword",
                    name=keyword.token,
                    idf=keyword.idf,
                )
                self._bidirectional_edge(
                    chunk.chunk_id,
                    keyword_id,
                    "HAS_KEYWORD",
                    "KEYWORD_OF",
                    weight=keyword.score,
                    confidence=min(1.0, keyword.score),
                )

        self._annotate_and_share_entities(entity_chunks, entities, len(chunks))
        self._annotate_and_share_keywords(keyword_chunks, keyword_scores)
        self._add_code_relations(symbols, element_chunks)

    def _add_hierarchy(self, document_id: str, chunk: Chunk) -> None:
        parent_id = document_id
        for depth in range(1, len(chunk.section_path) + 1):
            section_path = chunk.section_path[:depth]
            section_id = stable_id(
                "section",
                chunk.source_id,
                chunk.source_version,
                *section_path,
            )
            self.graph.add_node(
                section_id,
                kind="section",
                source_id=chunk.source_id,
                source_version=chunk.source_version,
                name=section_path[-1],
                path=list(section_path),
                depth=depth,
            )
            self._bidirectional_edge(
                parent_id,
                section_id,
                "CONTAINS_SECTION",
                "PART_OF",
                weight=1.0,
                confidence=1.0,
            )
            parent_id = section_id
        self._bidirectional_edge(
            parent_id,
            chunk.chunk_id,
            "CONTAINS_CHUNK",
            "PART_OF",
            weight=1.0,
            confidence=1.0,
        )

    def _annotate_and_share_entities(
        self,
        entity_chunks: dict[str, set[str]],
        entities: dict[str, Entity],
        chunk_count: int,
    ) -> None:
        total_chunks = max(1, chunk_count)
        for entity_id, related in entity_chunks.items():
            idf = math.log((1 + total_chunks) / (1 + len(related))) + 1
            self.graph.nodes[entity_id]["idf"] = idf
            self.graph.nodes[entity_id]["frequency"] = len(related)
            if not 1 < len(related) <= self.shared_relation_limit:
                continue
            confidence = entities[entity_id].confidence
            for source, target in combinations(sorted(related), 2):
                self._bidirectional_edge(
                    source,
                    target,
                    f"SHARES_ENTITY:{entity_id}",
                    f"SHARES_ENTITY:{entity_id}",
                    kind="SHARES_ENTITY",
                    entity_id=entity_id,
                    weight=idf,
                    confidence=confidence,
                )

    def _annotate_and_share_keywords(
        self,
        keyword_chunks: dict[str, set[str]],
        scores: dict[tuple[str, str], KeywordScore],
    ) -> None:
        for keyword_id, related in keyword_chunks.items():
            self.graph.nodes[keyword_id]["frequency"] = len(related)
            if not 1 < len(related) <= self.shared_relation_limit:
                continue
            for source, target in combinations(sorted(related), 2):
                weight = min(scores[(source, keyword_id)].score, scores[(target, keyword_id)].score)
                self._bidirectional_edge(
                    source,
                    target,
                    f"SHARES_KEYWORD:{keyword_id}",
                    f"SHARES_KEYWORD:{keyword_id}",
                    kind="SHARES_KEYWORD",
                    keyword_id=keyword_id,
                    weight=weight,
                    confidence=min(1.0, weight),
                )

    def _add_code_relations(
        self,
        symbols: Sequence[CodeSymbol],
        element_chunks: dict[str, set[str]],
    ) -> None:
        symbol_ids = {symbol.symbol_id for symbol in symbols}
        symbol_lookup: dict[str, str] = {}
        for symbol in symbols:
            symbol_lookup.setdefault(symbol.name.casefold(), symbol.symbol_id)
            symbol_lookup.setdefault(symbol.qualified_name.casefold(), symbol.symbol_id)
            self.graph.add_node(
                symbol.symbol_id,
                kind="symbol",
                symbol_kind=symbol.kind,
                name=symbol.name,
                qualified_name=symbol.qualified_name,
                language=symbol.language,
                line_start=symbol.line_start,
                line_end=symbol.line_end,
            )
            for chunk_id in sorted(element_chunks.get(symbol.element_id, ())):
                self._bidirectional_edge(
                    chunk_id,
                    symbol.symbol_id,
                    "DEFINES",
                    "DEFINED_IN",
                    weight=1.0,
                    confidence=1.0,
                )
            if symbol.parent_symbol_id in symbol_ids:
                self._bidirectional_edge(
                    symbol.parent_symbol_id,
                    symbol.symbol_id,
                    "CONTAINS_SYMBOL",
                    "MEMBER_OF",
                    weight=1.0,
                    confidence=1.0,
                )
        for symbol in symbols:
            for imported in symbol.imports:
                target = self._symbol_target(imported, symbol_lookup, "module")
                self.graph.add_edge(
                    symbol.symbol_id,
                    target,
                    key=f"IMPORTS:{target}",
                    kind="IMPORTS",
                    weight=1.0,
                    confidence=1.0,
                )
            for called in symbol.calls:
                target = self._symbol_target(called, symbol_lookup, "external_symbol")
                self.graph.add_edge(
                    symbol.symbol_id,
                    target,
                    key=f"CALLS:{target}",
                    kind="CALLS",
                    weight=1.0,
                    confidence=1.0,
                )

    def _symbol_target(
        self,
        name: str,
        lookup: dict[str, str],
        external_kind: str,
    ) -> str:
        folded = name.casefold()
        target = lookup.get(folded) or lookup.get(folded.rsplit(".", 1)[-1])
        if target:
            return target
        target = stable_id("external-symbol", external_kind, folded)
        self.graph.add_node(target, kind=external_kind, name=name)
        return target

    def _bidirectional_edge(
        self,
        source: str,
        target: str,
        forward_key: str,
        reverse_key: str,
        *,
        kind: str | None = None,
        weight: float,
        confidence: float,
        **metadata: object,
    ) -> None:
        self.graph.add_edge(
            source,
            target,
            key=forward_key,
            kind=kind or forward_key.split(":", 1)[0],
            weight=weight,
            confidence=confidence,
            **metadata,
        )
        self.graph.add_edge(
            target,
            source,
            key=reverse_key,
            kind=kind or reverse_key.split(":", 1)[0],
            weight=weight,
            confidence=confidence,
            **metadata,
        )

    @staticmethod
    def _document_id(chunk: Chunk) -> str:
        return stable_id("document", chunk.source_id, chunk.source_version)

    def add_relations(self, relations: Iterable[Relation]) -> None:
        for relation in relations:
            self.graph.add_edge(
                relation.source_id,
                relation.target_id,
                key=relation.kind,
                kind=relation.kind,
                weight=relation.weight,
                confidence=relation.confidence,
                provenance=list(relation.provenance),
                **relation.metadata,
            )

    def add_semantic_edges(
        self,
        chunk_ids: Sequence[str],
        dense_vectors: Sequence[Sequence[float]] | np.ndarray,
        top_k: int = 3,
        threshold: float = 0.72,
        block_size: int = 256,
    ) -> None:
        if top_k < 1:
            raise ValueError("Semantic top-k must be positive")
        if block_size < 1:
            raise ValueError("Semantic block size must be positive")
        vectors = np.asarray(dense_vectors, dtype=np.float32)
        if not chunk_ids:
            return
        if vectors.ndim != 2 or vectors.shape[0] != len(chunk_ids):
            raise ValueError("Chunk IDs and dense vectors must have equal row counts")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = np.divide(
            vectors,
            norms,
            out=np.zeros_like(vectors),
            where=norms > 0,
        )
        nearest: list[dict[int, float]] = []
        count = len(chunk_ids)
        for start in range(0, count, block_size):
            similarities = normalized[start : start + block_size] @ normalized.T
            for local_index, row in enumerate(similarities):
                source_index = start + local_index
                row[source_index] = -np.inf
                eligible = np.flatnonzero(row >= threshold)
                if len(eligible) > top_k:
                    selected = eligible[np.argpartition(row[eligible], -top_k)[-top_k:]]
                else:
                    selected = eligible
                nearest.append(
                    {
                        int(target): float(row[target])
                        for target in sorted(selected, key=lambda target: -row[target])
                    }
                )
        for source_index, targets in enumerate(nearest):
            for target_index, similarity in targets.items():
                if source_index >= target_index or source_index not in nearest[target_index]:
                    continue
                self._bidirectional_edge(
                    chunk_ids[source_index],
                    chunk_ids[target_index],
                    "SEMANTIC_SIMILAR",
                    "SEMANTIC_SIMILAR",
                    weight=similarity,
                    confidence=similarity,
                )

    def expand(
        self,
        seed_scores: dict[str, float],
        max_hops: int = 2,
        max_nodes: int = 160,
        per_node_limit: int = 12,
    ) -> dict[str, tuple[float, tuple[str, ...]]]:
        if not seed_scores:
            return {}
        allowed_seeds = {node: score for node, score in seed_scores.items() if node in self.graph}
        if not allowed_seeds:
            return {}
        frontier: list[tuple[str, float, tuple[str, ...], int]] = [
            (node, score, (node,), 0) for node, score in allowed_seeds.items()
        ]
        best: dict[str, tuple[float, tuple[str, ...]]] = {
            node: (score, (node,)) for node, score in allowed_seeds.items()
        }
        while frontier and len(best) < max_nodes:
            node, score, path, hops = frontier.pop(0)
            if hops >= max_hops:
                continue
            neighbours: list[tuple[str, float]] = []
            for target, edges in self.graph.adj[node].items():
                edge_score = max(
                    float(data.get("weight", 1.0)) * float(data.get("confidence", 1.0))
                    for data in edges.values()
                )
                if self.graph.nodes[target].get("kind") in {"entity", "keyword"}:
                    edge_score *= min(2.0, float(self.graph.nodes[target].get("idf", 1.0)))
                neighbours.append((target, edge_score))
            ordered_neighbours = sorted(
                neighbours,
                key=lambda item: (
                    self.graph.nodes[item[0]].get("kind") != "chunk",
                    -item[1],
                ),
            )
            for target, edge_score in ordered_neighbours[:per_node_limit]:
                if target in path:
                    continue
                # 扩展只做衰减传播：边权封顶 1.0 再乘 hop 衰减，
                # 保证邻居分数永远低于其种子，图扩展无法把词法强命中挤出重排窗口。
                decayed_edge = min(1.0, edge_score)
                candidate = score * decayed_edge * (0.68 ** (hops + 1))
                candidate_path = (*path, target)
                if candidate > best.get(target, (0.0, ()))[0]:
                    best[target] = (candidate, candidate_path)
                    frontier.append((target, candidate, candidate_path, hops + 1))
                if len(best) >= max_nodes:
                    break
        return {
            node: value
            for node, value in best.items()
            if self.graph.nodes[node].get("kind") == "chunk"
        }

    def query_subgraph(
        self,
        nodes: Iterable[str],
        radius: int = 1,
        limit: int = 120,
    ) -> nx.MultiDiGraph:
        selected: set[str] = set()
        for node in nodes:
            if node not in self.graph:
                continue
            selected.add(node)
            selected.update(
                list(nx.single_source_shortest_path_length(self.graph, node, cutoff=radius))[:limit]
            )
            if len(selected) >= limit:
                break
        return self.graph.subgraph(list(selected)[:limit]).copy()

    def save(self, path: Path) -> str:
        payload = nx.node_link_data(self.graph, edges="edges")
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        with gzip.open(temporary, "wb") as stream:
            stream.write(encoded)
        temporary.replace(path)
        return sha256_bytes(encoded)

    @classmethod
    def load(cls, path: Path, extractor: EntityExtractor | None = None) -> KnowledgeGraph:
        with gzip.open(path, "rb") as stream:
            payload = json.loads(stream.read().decode("utf-8"))
        instance = cls(extractor)
        instance.graph = nx.node_link_graph(payload, edges="edges", directed=True, multigraph=True)
        return instance
