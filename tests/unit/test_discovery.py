from __future__ import annotations

from pathlib import Path

import pytest

from secknow.domain.ids import sha256_bytes, stable_id
from secknow.ingestion.discovery import DocumentDiscovery, UnsafeCorpusPath


def test_stable_id_is_deterministic() -> None:
    assert stable_id("chunk", "a", 1) == stable_id("chunk", "a", 1)
    assert stable_id("chunk", "a", 1) != stable_id("chunk", "a", 2)


def test_discovery_rejects_path_outside_root(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    discovery = DocumentDiscovery(corpus)
    with pytest.raises(UnsafeCorpusPath):
        discovery.inspect(outside)


def test_discovery_inspects_supported_text(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    path = corpus / "note.md"
    path.write_text("# 栈溢出\n\nROP 链。", encoding="utf-8")
    document = DocumentDiscovery(corpus).inspect(path)
    assert document.relative_path == "note.md"
    assert document.sha256 == sha256_bytes(path.read_bytes())
    assert document.format.value == "md"
