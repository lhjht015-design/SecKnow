from __future__ import annotations

from pathlib import Path

import pytest

from secknow.answer.llm import LocalLlmClient
from secknow.config import RuntimeMode, Settings


def settings(tmp_path: Path, url: str) -> Settings:
    return Settings(
        mode=RuntimeMode.LLM,
        llm_enabled=True,
        llm_base_url=url,
        corpus_root=tmp_path,
        data_root=tmp_path / "data",
        database_path=tmp_path / "data.db",
        snapshot_root=tmp_path / "snapshots",
        model_root=tmp_path / "models",
        source_manifest=tmp_path / "sources.yaml",
        bge_model_path=tmp_path / "bge",
        reranker_model_path=tmp_path / "reranker",
    )


def test_llm_endpoint_allows_localhost(tmp_path: Path) -> None:
    LocalLlmClient(settings(tmp_path, "http://127.0.0.1:11434/v1")).validate_endpoint()


def test_llm_endpoint_rejects_public_cloud(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        LocalLlmClient(settings(tmp_path, "https://8.8.8.8/v1")).validate_endpoint()
