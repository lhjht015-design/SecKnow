from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import networkx as nx


class Neo4jStore:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self._driver: Any = None

    def connect(self) -> None:
        if self._driver is not None:
            return
        try:
            from neo4j import GraphDatabase
        except ImportError as error:
            raise RuntimeError("Install SecKnow with the 'semantic' extra for Neo4j") from error
        self._driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            connection_timeout=3.0,
            connection_acquisition_timeout=3.0,
        )

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def healthy(self) -> bool:
        try:
            self.connect()
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    def initialize(self) -> None:
        self.connect()
        statements = (
            "CREATE CONSTRAINT secknow_node_snapshot IF NOT EXISTS "
            "FOR (n:SecKnowNode) REQUIRE n.snapshot_key IS UNIQUE",
            "CREATE CONSTRAINT secknow_graph_name IF NOT EXISTS "
            "FOR (n:SecKnowGraph) REQUIRE n.name IS UNIQUE",
            "CREATE INDEX secknow_node_lookup IF NOT EXISTS "
            "FOR (n:SecKnowNode) ON (n.graph_version, n.id)",
        )
        with self._driver.session() as session:
            for statement in statements:
                session.run(statement).consume()

    def replace_graph(
        self,
        graph: nx.MultiDiGraph,
        version: str,
        *,
        batch_size: int = 1000,
    ) -> None:
        if not version:
            raise ValueError("Graph version must not be empty")
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        self.initialize()
        with self._driver.session() as session:
            for rows in self._batches(self._node_rows(graph, version), batch_size):
                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (n:SecKnowNode {snapshot_key: row.snapshot_key})
                    SET n = row.properties
                    SET n.snapshot_key = row.snapshot_key,
                        n.id = row.id,
                        n.graph_version = $version
                    """,
                    rows=rows,
                    version=version,
                ).consume()
            for rows in self._batches(self._edge_rows(graph, version), batch_size):
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (source:SecKnowNode {snapshot_key: row.source_key})
                    MATCH (target:SecKnowNode {snapshot_key: row.target_key})
                    MERGE (source)-[r:RELATED {
                        graph_version: $version,
                        edge_key: row.edge_key
                    }]->(target)
                    SET r = row.properties
                    SET r.graph_version = $version,
                        r.edge_key = row.edge_key
                    """,
                    rows=rows,
                    version=version,
                ).consume()
            session.run(
                """
                MERGE (catalog:SecKnowGraph {name: 'active'})
                SET catalog.active_version = $version
                """,
                version=version,
            ).consume()
            session.run(
                """
                MATCH (n:SecKnowNode)
                WHERE n.graph_version <> $version
                DETACH DELETE n
                """,
                version=version,
            ).consume()

    def neighbourhood(
        self,
        seed_ids: Sequence[str],
        max_hops: int = 2,
        limit: int = 160,
    ) -> dict[str, Any]:
        if not seed_ids:
            return {"nodes": [], "edges": []}
        self.connect()
        safe_hops = max(1, min(4, max_hops))
        query = f"""
            MATCH (catalog:SecKnowGraph {{name: 'active'}})
            MATCH path=(seed:SecKnowNode)-[:RELATED*1..{safe_hops}]-(related:SecKnowNode)
            WHERE seed.graph_version = catalog.active_version
              AND seed.id IN $seed_ids
              AND all(node IN nodes(path)
                      WHERE node.graph_version = catalog.active_version)
            WITH path LIMIT $limit
            UNWIND nodes(path) AS n
            UNWIND relationships(path) AS r
            RETURN collect(DISTINCT properties(n)) AS nodes,
                   collect(DISTINCT {{source: startNode(r).id, target: endNode(r).id,
                                      kind: r.kind, weight: r.weight,
                                      confidence: r.confidence}}) AS edges
        """
        with self._driver.session() as session:
            record = session.run(query, seed_ids=list(seed_ids), limit=limit).single()
        return dict(record) if record else {"nodes": [], "edges": []}

    @classmethod
    def _node_rows(
        cls,
        graph: nx.MultiDiGraph,
        version: str,
    ) -> Iterator[dict[str, Any]]:
        for node_id, attributes in graph.nodes(data=True):
            identifier = str(node_id)
            yield {
                "id": identifier,
                "snapshot_key": f"{version}:{identifier}",
                "properties": cls._properties(attributes),
            }

    @classmethod
    def _edge_rows(
        cls,
        graph: nx.MultiDiGraph,
        version: str,
    ) -> Iterator[dict[str, Any]]:
        for source, target, key, attributes in graph.edges(keys=True, data=True):
            yield {
                "source_key": f"{version}:{source}",
                "target_key": f"{version}:{target}",
                "edge_key": str(key),
                "properties": cls._properties(attributes),
            }

    @staticmethod
    def _properties(attributes: Mapping[str, object]) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        for key, value in attributes.items():
            if value is None or key in {"id", "snapshot_key", "graph_version"}:
                continue
            if isinstance(value, (str, bool, int, float)):
                properties[key] = value
                continue
            if isinstance(value, (list, tuple)) and all(
                item is None or isinstance(item, (str, bool, int, float)) for item in value
            ):
                properties[key] = [item for item in value if item is not None]
                continue
            properties[key] = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        return properties

    @staticmethod
    def _batches(
        rows: Iterable[dict[str, Any]],
        batch_size: int,
    ) -> Iterator[list[dict[str, Any]]]:
        batch: list[dict[str, Any]] = []
        for row in rows:
            batch.append(row)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
