from __future__ import annotations

from pathlib import Path

import networkx as nx

from secknow.config import RuntimeMode, Settings
from secknow.domain.models import ElementKind, ParsedElement
from secknow.graph.builder import KnowledgeGraph
from secknow.ingestion.pipeline import IngestionPipeline, _quarantine_warnings
from secknow.retrieval.service import RetrievalService


def test_plain_pipeline_and_lexical_search(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    data = tmp_path / "data"
    corpus.mkdir()
    source = corpus / "pwn.txt"
    source.write_text(
        "栈溢出会覆盖函数返回地址。ROP 使用已有指令片段构造调用链。\n"
        "CVE-2024-1234 是一个示例漏洞编号。",
        encoding="utf-8",
    )
    settings = Settings(
        mode=RuntimeMode.STRICT_LEXICAL,
        corpus_root=corpus,
        data_root=data,
        database_path=data / "secknow.db",
        snapshot_root=data / "snapshots",
        model_root=tmp_path / "models",
        source_manifest=tmp_path / "sources.yaml",
        bge_model_path=tmp_path / "models" / "bge-m3",
        reranker_model_path=tmp_path / "models" / "reranker",
        chunk_size=128,
        chunk_overlap=16,
    )
    first = IngestionPipeline(settings).ingest_path(source)
    second = IngestionPipeline(settings).ingest_path(source)
    assert first.document.version_id == second.document.version_id
    assert [chunk.chunk_id for chunk in first.chunks] == [chunk.chunk_id for chunk in second.chunks]

    response = RetrievalService(settings).search("CVE-2024-1234")
    assert response.hits
    assert "CVE-2024-1234" in response.hits[0].chunk.text
    assert response.mode == "strict-lexical"


def test_quarantine_warnings_count_pages_not_diagnostics() -> None:
    element = ParsedElement(
        element_id="element:bad",
        source_id="document:test",
        source_version="version:test",
        kind=ElementKind.PARAGRAPH,
        text="damaged text",
        order=0,
        page=7,
        parser="pymupdf",
        diagnostics=("font-map-corruption", "font-map-corruption", "ocr-not-configured"),
        metadata={"disposition": "quarantine"},
    )

    assert _quarantine_warnings((element,)) == [
        "page:7:font-map-corruption|ocr-not-configured"
    ]


def test_pipeline_graph_keeps_symbols_from_other_active_documents(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    data = tmp_path / "data"
    corpus.mkdir()
    first_source = corpus / "parser.py"
    second_source = corpus / "helper.py"
    first_source.write_text("def parse_elf(data):\n    return data[:4]\n", encoding="utf-8")
    second_source.write_text(
        "def validate_header(data):\n    return bool(data)\n",
        encoding="utf-8",
    )
    settings = Settings(
        mode=RuntimeMode.STRICT_LEXICAL,
        corpus_root=corpus,
        data_root=data,
        database_path=data / "secknow.db",
        snapshot_root=data / "snapshots",
        model_root=tmp_path / "models",
        source_manifest=tmp_path / "sources.yaml",
        bge_model_path=tmp_path / "models" / "bge-m3",
        reranker_model_path=tmp_path / "models" / "reranker",
        chunk_size=128,
        chunk_overlap=16,
    )
    pipeline = IngestionPipeline(settings)
    first = pipeline.ingest_path(first_source)
    second = pipeline.ingest_path(second_source)

    graph_path = data / "graph" / f"{second.graph_version}.json.gz"
    graph = KnowledgeGraph.load(graph_path).graph
    symbol_names = {
        attributes["name"]
        for _, attributes in graph.nodes(data=True)
        if attributes.get("kind") == "symbol"
    }
    assert symbol_names == {"parse_elf", "validate_header"}
    active_symbols = pipeline.store.active_code_symbols()
    assert {symbol.name for symbol in active_symbols} == symbol_names
    assert isinstance(graph, nx.MultiDiGraph)
    assert first.symbols[0].symbol_id in graph
