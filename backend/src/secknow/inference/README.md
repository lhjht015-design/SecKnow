# 4.4 Inference

`secknow.inference` is the 4.4 orchestration layer.

It sits between:

- 4.5 API: receives user-facing query requests
- 4.3 vector_store: provides dense / hybrid retrieval and baseline access
- 4.1 text_processing: provides the embedding path reused for query encoding

## Scope

4.4 is responsible for:

- semantic retrieval orchestration
- code risk scan orchestration
- query encoding reuse
- result reranking
- result formatting

4.4 is not responsible for:

- document ingestion and chunk generation
- vector persistence implementation
- direct HTTP routing

## Current directory layout

```text
inference/
├── __init__.py
├── README.md
├── config.py
├── exceptions.py
├── facade.py
├── models.py
├── retrieval/
├── scanners/
├── schemas/
├── services/
└── tests/
```

## Responsibilities by file

- `facade.py`
  - public 4.4 entrypoints for 4.5
  - keeps API layer away from internal service wiring
- `config.py`
  - default top-k, rerank weights, supported languages
- `models.py`
  - internal dataclasses shared inside 4.4
- `schemas/*`
  - request / response contracts consumed by 4.5 and 4.6
- `services/query_encoder.py`
  - reuses 4.1 embedding path for query encoding
- `services/semantic_search.py`
  - query -> encode -> retrieve -> rerank -> format
- `services/code_risk_scan.py`
  - code risk scan workflow
- `services/reranker.py`
  - local result reranking policy
- `services/formatter.py`
  - normalize raw search hits into UI/API friendly output
- `retrieval/semantic.py`
  - thin adapter over 4.3 `search()` / `hybrid_search()`
- `retrieval/baseline.py`
  - thin adapter over 4.3 `get_baseline()`
- `scanners/router.py`
  - picks language-specific code scanner
- `scanners/text_fallback.py`
  - first-stage fallback scanner without AST dependencies
- `scanners/treesitter_scan.py`
  - reserved AST-based scanner entrypoint

## Intended public APIs

- `semantic_search(query, zone_id, top_k=10, filters=None)`
- `code_risk_scan(code_snippet, lang, zone_id="cyber", top_k=10)`

## Current implementation stage

This directory is scaffolded as the initial 4.4 baseline.

Implemented now:

- package layout
- schema definitions
- service wiring
- semantic search skeleton
- code scan skeleton

Planned next:

1. wire a concrete API layer in 4.5
2. tune rerank policy against real retrieval samples
3. replace fallback code scanning with tree-sitter based slicing
4. add integration tests against offline vector store
