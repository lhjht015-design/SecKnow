from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import networkx as nx
import pytest

from secknow.stores.milvus import MilvusStore
from secknow.stores.neo4j import Neo4jStore


class FakeAnnSearchRequest:
    def __init__(
        self,
        *,
        data: list[object],
        anns_field: str,
        param: dict[str, object],
        limit: int,
    ) -> None:
        self.data = data
        self.anns_field = anns_field
        self.param = param
        self.limit = limit


class FakeWeightedRanker:
    def __init__(self, *weights: float) -> None:
        self.weights = weights


class FakeMilvusClient:
    def __init__(self) -> None:
        self.hybrid_calls: list[dict[str, Any]] = []

    @staticmethod
    def has_collection(collection: str) -> bool:
        return collection == "secknow_chunks"

    def hybrid_search(self, **kwargs: Any) -> list[list[dict[str, object]]]:
        self.hybrid_calls.append(kwargs)
        return [[{"entity": {"chunk_id": "hybrid-hit"}, "distance": 0.88}]]

    @staticmethod
    def search(**kwargs: Any) -> list[list[dict[str, object]]]:
        field = kwargs["anns_field"]
        return [[{"entity": {"chunk_id": field}, "distance": 0.7}]]


def test_milvus_uses_server_side_hybrid_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "pymilvus",
        SimpleNamespace(
            AnnSearchRequest=FakeAnnSearchRequest,
            WeightedRanker=FakeWeightedRanker,
        ),
    )
    client = FakeMilvusClient()
    store = MilvusStore("http://localhost:19530")
    store._client = client

    result = store.hybrid_search(
        [0.1, 0.2],
        {1: 0.5},
        top_k=5,
        dense_weight=0.8,
        sparse_weight=1.2,
    )

    assert [hit.chunk_id for hit in result["hybrid"]] == ["hybrid-hit"]
    assert len(client.hybrid_calls) == 1
    call = client.hybrid_calls[0]
    assert len(call["reqs"]) == 2
    assert call["ranker"].weights == pytest.approx((2 / 3, 1.0))


def test_milvus_rejects_empty_hybrid_weight() -> None:
    store = MilvusStore("http://localhost:19530")
    store._client = FakeMilvusClient()
    with pytest.raises(ValueError, match="not both zero"):
        store.hybrid_search([0.1], {1: 0.5}, dense_weight=0.0, sparse_weight=0.0)


class FakeResult:
    @staticmethod
    def single() -> dict[str, list[object]]:
        return {"nodes": [], "edges": []}

    def consume(self) -> FakeResult:
        return self


class FakeSession:
    def __init__(self) -> None:
        self.query = ""
        self.parameters: dict[str, object] = {}
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def run(self, query: str, **parameters: object) -> FakeResult:
        self.query = query
        self.parameters = parameters
        self.calls.append((query, parameters))
        return FakeResult()


class FakeDriver:
    def __init__(self) -> None:
        self.current_session = FakeSession()

    def session(self) -> FakeSession:
        return self.current_session


def test_neo4j_neighbourhood_matches_seed_variable() -> None:
    driver = FakeDriver()
    store = Neo4jStore("bolt://localhost:7687", "neo4j", "password")
    store._driver = driver

    assert store.neighbourhood(["chunk:1"], max_hops=2) == {"nodes": [], "edges": []}
    assert "MATCH path=(seed:SecKnowNode)-[:RELATED*1..2]" in driver.current_session.query
    assert "{id: seed_id}" not in driver.current_session.query
    assert driver.current_session.parameters["seed_ids"] == ["chunk:1"]


def test_neo4j_replaces_managed_graph_after_atomic_activation() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("chunk:1", kind="chunk", section=["Stack"])
    graph.add_node("entity:1", kind="entity", name="CVE-2024-1234")
    graph.add_edge(
        "chunk:1",
        "entity:1",
        key="MENTIONS",
        kind="MENTIONS",
        weight=1.0,
        confidence=0.9,
    )
    driver = FakeDriver()
    store = Neo4jStore("bolt://localhost:7687", "neo4j", "password")
    store._driver = driver

    store.replace_graph(graph, "graph-v1", batch_size=1)

    calls = driver.current_session.calls
    node_calls = [call for call in calls if "MERGE (n:SecKnowNode" in call[0]]
    edge_calls = [call for call in calls if "MERGE (source)-[r:RELATED" in call[0]]
    activation_index = next(
        index for index, call in enumerate(calls) if "active_version = $version" in call[0]
    )
    cleanup_index = next(
        index for index, call in enumerate(calls) if "DETACH DELETE n" in call[0]
    )
    assert len(node_calls) == 2
    assert len(edge_calls) == 1
    assert activation_index > calls.index(edge_calls[0])
    assert cleanup_index > activation_index
    node_rows = node_calls[0][1]["rows"]
    edge_rows = edge_calls[0][1]["rows"]
    assert isinstance(node_rows, list)
    assert isinstance(edge_rows, list)
    assert node_rows[0]["snapshot_key"].startswith("graph-v1:")
    assert edge_rows[0]["edge_key"] == "MENTIONS"
