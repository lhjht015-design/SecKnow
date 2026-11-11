from __future__ import annotations

import numpy as np

from secknow.domain.ids import sha256_bytes, stable_id
from secknow.domain.models import Chunk, Citation, CodeSymbol
from secknow.graph.builder import KnowledgeGraph


def chunk(
    identifier: str,
    text: str,
    previous: str | None = None,
    *,
    source_id: str = "doc",
    section_path: tuple[str, ...] = (),
    element_id: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=identifier,
        source_id=source_id,
        source_version="version",
        text=text,
        content_sha256=sha256_bytes(text.encode()),
        element_ids=(element_id or f"element:{identifier}",),
        order=int(identifier[-1]),
        token_count=5,
        section_path=section_path,
        previous_chunk_id=previous,
        citation=Citation(
            source_id=source_id,
            source_version="version",
            title="Graph test",
        ),
    )


def test_graph_expands_through_shared_entity_with_bounds() -> None:
    chunks = (
        chunk("chunk:0", "CVE-2024-1234 affects an ELF parser"),
        chunk("chunk:1", "CVE-2024-1234 can lead to memory corruption", "chunk:0"),
        chunk("chunk:2", "unrelated cryptography notes", "chunk:1"),
    )
    graph = KnowledgeGraph()
    graph.build(chunks)
    expanded = graph.expand({"chunk:0": 1.0}, max_hops=2, max_nodes=5)
    assert "chunk:1" in expanded
    assert len(expanded) <= 5
    assert len(expanded["chunk:1"][1]) <= 3


def test_graph_builds_hierarchy_keywords_and_code_relations() -> None:
    chunks = (
        chunk(
            "chunk:0",
            "CVE-2024-1234 parser calls helper and validates ELF headers",
            section_path=("Reverse engineering", "ELF"),
            element_id="element:code",
        ),
        chunk(
            "chunk:1",
            "CVE-2024-1234 corrupts parser memory in ELF processing",
            "chunk:0",
            section_path=("Reverse engineering", "ELF"),
        ),
    )
    parent = CodeSymbol(
        symbol_id="symbol:parser",
        element_id="element:code",
        name="Parser",
        qualified_name="Parser",
        kind="class",
        language="python",
        line_start=1,
        line_end=5,
    )
    method = CodeSymbol(
        symbol_id="symbol:parse",
        element_id="element:code",
        name="parse",
        qualified_name="Parser.parse",
        kind="function",
        language="python",
        line_start=2,
        line_end=5,
        parent_symbol_id=parent.symbol_id,
        imports=("struct",),
        calls=("helper",),
    )

    graph = KnowledgeGraph()
    graph.build(chunks, (parent, method))

    document_id = stable_id("document", "doc", "version")
    section_id = stable_id("section", "doc", "version", "Reverse engineering")
    cve_id = stable_id("entity", "vulnerability", "CVE-2024-1234")
    assert graph.graph.has_edge(document_id, section_id, "CONTAINS_SECTION")
    assert graph.graph.has_edge("chunk:0", cve_id, "MENTIONS")
    assert any(
        data.get("kind") == "HAS_KEYWORD"
        for _, _, data in graph.graph.edges("chunk:0", data=True)
    )
    assert any(
        data.get("kind") == "SHARES_ENTITY"
        for _, _, data in graph.graph.edges("chunk:0", data=True)
    )
    assert graph.graph.has_edge("chunk:0", method.symbol_id, "DEFINES")
    assert graph.graph.has_edge(parent.symbol_id, method.symbol_id, "CONTAINS_SYMBOL")
    assert any(
        data.get("kind") == "IMPORTS"
        for _, _, data in graph.graph.edges(method.symbol_id, data=True)
    )
    assert any(
        data.get("kind") == "CALLS"
        for _, _, data in graph.graph.edges(method.symbol_id, data=True)
    )


def test_graph_limits_direct_shared_entity_edges_for_common_hubs() -> None:
    chunks = tuple(
        chunk(f"chunk:{index}", f"CVE-2024-9999 common record number {index}")
        for index in range(13)
    )
    graph = KnowledgeGraph(shared_relation_limit=12)
    graph.build(chunks)

    shared_edges = [
        data
        for _, _, data in graph.graph.edges(data=True)
        if data.get("kind") == "SHARES_ENTITY"
    ]
    assert shared_edges == []


def test_semantic_edges_require_reciprocal_top_k() -> None:
    chunks = (
        chunk("chunk:0", "alpha"),
        chunk("chunk:1", "beta"),
        chunk("chunk:2", "gamma"),
    )
    dense = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.1],
            [0.8, 0.6],
        ],
        dtype=np.float32,
    )
    graph = KnowledgeGraph()
    graph.build(chunks)
    graph.add_semantic_edges(
        [item.chunk_id for item in chunks],
        dense,
        top_k=1,
        threshold=0.5,
        block_size=2,
    )

    semantic_edges = [
        (source, target)
        for source, target, data in graph.graph.edges(data=True)
        if data.get("kind") == "SEMANTIC_SIMILAR"
    ]
    assert semantic_edges == [("chunk:0", "chunk:1"), ("chunk:1", "chunk:0")]
