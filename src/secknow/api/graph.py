from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
from fastapi import APIRouter, HTTPException, Query

from secknow.api.schemas import SettingsDependency
from secknow.graph.builder import KnowledgeGraph

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/{chunk_id}")
def graph_for_chunk(
    chunk_id: str,
    settings: SettingsDependency,
    radius: int = Query(default=1, ge=0, le=3),
    limit: int = Query(default=80, ge=1, le=200),
) -> dict[str, Any]:
    version = _active_graph_version(settings.database_path)
    if not version:
        raise HTTPException(status_code=404, detail="graph snapshot is not available")
    snapshot = settings.data_root / "graph" / f"{version}.json.gz"
    if not snapshot.exists():
        raise HTTPException(status_code=404, detail="graph snapshot is missing")
    try:
        graph = KnowledgeGraph.load(snapshot)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail="graph snapshot is unreadable") from error
    if chunk_id not in graph.graph:
        raise HTTPException(status_code=404, detail="chunk is not present in the graph")
    subgraph = graph.query_subgraph((chunk_id,), radius=radius, limit=limit)
    return _serialize(subgraph)


def _active_graph_version(database_path: Path) -> str | None:
    from secknow.stores.metadata import MetadataStore

    value = MetadataStore(database_path).get_setting("active_graph_version")
    return str(value) if value else None


def _serialize(graph: nx.MultiDiGraph) -> dict[str, Any]:
    nodes = [
        {"id": str(node), **{key: value for key, value in data.items() if _json_value(value)}}
        for node, data in graph.nodes(data=True)
    ]
    edges = [
        {
            "id": f"{source}:{target}:{key}",
            "source": str(source),
            "target": str(target),
            **{name: value for name, value in data.items() if _json_value(value)},
        }
        for source, target, key, data in graph.edges(keys=True, data=True)
    ]
    return {"nodes": nodes, "edges": edges}


def _json_value(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool, list, tuple))
