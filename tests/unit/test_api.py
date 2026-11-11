from __future__ import annotations

from pathlib import Path

import fitz
from fastapi.testclient import TestClient

from secknow.api.app import create_app
from secknow.config import RuntimeMode, Settings
from secknow.ingestion.document_parser import PdfExtractor
from secknow.ingestion.pipeline import IngestionPipeline
from secknow.stores.metadata import MetadataStore


def settings_for(tmp_path: Path) -> Settings:
    corpus = tmp_path / "corpus"
    data = tmp_path / "data"
    corpus.mkdir()
    return Settings(
        mode=RuntimeMode.STRICT_LEXICAL,
        corpus_root=corpus,
        data_root=data,
        database_path=data / "secknow.db",
        snapshot_root=data / "snapshots",
        model_root=tmp_path / "models",
        frontend_dist=tmp_path / "dist",
        source_manifest=tmp_path / "sources" / "sources.yaml",
        bge_model_path=tmp_path / "models" / "bge-m3",
        reranker_model_path=tmp_path / "models" / "reranker",
        chunk_size=128,
        chunk_overlap=16,
    )


def test_health_config_and_safe_index_job(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    source = settings.corpus_root / "guide.txt"
    source.write_text("ROP 使用已有指令片段构造调用链。", encoding="utf-8")

    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["components"]["milvus"]["status"] == "disabled"

        config = client.get("/api/config").json()
        assert config["mode"] == "strict-lexical"
        assert "neo4j_password" not in config
        assert "llm_api_key" not in config

        accepted = client.post("/api/index", json={"path": "guide.txt"})
        duplicate = client.post("/api/index", json={"path": "guide.txt"})
        assert accepted.status_code == 202
        assert duplicate.status_code == 202
        assert duplicate.json()["job_id"] == accepted.json()["job_id"]
        job = client.get(f"/api/jobs/{accepted.json()['job_id']}").json()
        assert job["status"] == "queued"
        assert job["payload"]["path"] == "guide.txt"

        escaped = client.post("/api/index", json={"path": "../outside.txt"})
        assert escaped.status_code == 422


def test_upload_accepts_multiple_formats_and_enqueues_one_batch_job(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/documents/upload",
            files=[
                ("files", ("notes.md", b"# ROP\nUse gadgets.", "text/markdown")),
                ("files", ("parser.py", b"def parse(data):\n    return data\n", "text/x-python")),
            ],
            data={"auto_index": "true"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "queued"
        assert body["job_id"]
        assert {item["format"] for item in body["files"]} == {"md", "py"}
        for item in body["files"]:
            assert (settings.corpus_root / item["relative_path"]).is_file()
            assert not item["relative_path"].startswith("../")
        job = client.get(f"/api/jobs/{body['job_id']}").json()
        assert job["payload"]["recursive"] is False
        assert len(job["payload"]["paths"]) == 2

        unsupported = client.post(
            "/api/documents/upload",
            files={"files": ("payload.exe", b"MZ", "application/octet-stream")},
        )
    assert unsupported.status_code == 415


def test_search_returns_hits_and_evidence_contract(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    source = settings.corpus_root / "pwn.txt"
    source.write_text(
        "栈溢出可能覆盖函数返回地址。ROP 使用已有指令片段构造调用链。",
        encoding="utf-8",
    )
    IngestionPipeline(settings).ingest_path(source)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/search",
            json={
                "query": "ROP",
                "mode": "strict-lexical",
                "top_k": 5,
                "use_llm": False,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "ROP"
    assert body["hits"]
    assert body["hits"][0]["chunk"]["citation"]["relative_path"] == "pwn.txt"
    assert body["answer"]["sections"]
    assert body["generated_answer"] is None
    assert body["ranking_score_calibrated"] is False


def test_search_request_cannot_raise_server_capability(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    source = settings.corpus_root / "boundary.txt"
    source.write_text("CVE-2026-12345 是用于验证模式边界的测试记录。", encoding="utf-8")
    IngestionPipeline(settings).ingest_path(source)

    with TestClient(create_app(settings)) as client:
        elevated = client.post(
            "/api/search",
            json={
                "query": "CVE-2026-12345",
                "mode": "competition",
                "top_k": 5,
                "use_llm": False,
            },
        )
        llm = client.post(
            "/api/search",
            json={
                "query": "CVE-2026-12345",
                "mode": "llm",
                "top_k": 5,
                "use_llm": True,
            },
        )

    assert elevated.status_code == 200
    assert elevated.json()["mode"] == "strict-lexical"
    assert not any("BGE-M3" in warning for warning in elevated.json()["warnings"])
    assert llm.status_code == 409
    assert llm.json()["detail"] == "local LLM generation is disabled"


def test_sources_endpoint_reports_governed_manifest(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    settings.source_manifest.parent.mkdir(parents=True)
    settings.source_manifest.write_text(
        """version: 1
sources:
  - id: cisa-kev
    name: CISA KEV
    description: Governed fixture catalog.
    homepage: https://www.cisa.gov/known-exploited-vulnerabilities-catalog
    adapter: cisa_kev_json
    risk_state: default
    license:
      expression: LicenseRef-US-Government-Work
      name: United States Government Work
      url: https://www.cisa.gov/about/website-policies
      attribution_required: false
      redistribution_allowed: true
    locations:
      - url: https://www.cisa.gov/known-exploited-vulnerabilities-catalog
        kind: index
        media_type: application/json
""",
        encoding="utf-8",
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/api/sources")
        missing = client.post(
            "/api/sources/snapshots/" + "a" * 64 + "/publish",
            json={"allow_quarantine": False},
        )
        invalid_snapshot = client.post(
            "/api/sources/snapshots/not-a-sha256/publish",
            json={"allow_quarantine": False},
        )
        invalid_source = client.post(
            "/api/sources/INVALID_SOURCE/rollback",
            json={"snapshot_id": None},
        )

    assert response.status_code == 200
    source = response.json()["sources"][0]
    assert source["id"] == "cisa-kev"
    assert source["risk_state"] == "default"
    assert source["license"]["expression"] == "LicenseRef-US-Government-Work"
    assert source["current_snapshot"] is None
    assert missing.status_code == 404
    assert invalid_snapshot.status_code == 422
    assert invalid_source.status_code == 422


def test_document_page_audit_reports_quality_and_attention_pages(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    pdf = settings.corpus_root / "audit.pdf"
    with fitz.open() as document:
        good = document.new_page(width=320, height=240)
        good.insert_text(
            (20, 60),
            "ROP uses existing instruction sequences and preserves citations. " * 4,
            fontsize=8,
        )
        document.new_page(width=320, height=240)
        document.save(pdf)

    pipeline = IngestionPipeline(settings)
    source = pipeline.discovery.inspect(pdf)
    pages = PdfExtractor().extract(source)
    elements = PdfExtractor().to_elements(source, pages)
    store = MetadataStore(settings.database_path)
    store.initialize()
    store.upsert_document(source, active=True)
    store.replace_elements(source.source_id, source.version_id, elements)
    store.activate_version(source.source_id, source.version_id)

    with TestClient(create_app(settings)) as client:
        audit = client.get(f"/api/documents/{source.source_id}/pages")
        attention = client.get(
            f"/api/documents/{source.source_id}/pages",
            params={"attention_only": True, "limit": 1},
        )
        missing = client.get("/api/documents/document:missing/pages")
        invalid_limit = client.get(
            f"/api/documents/{source.source_id}/pages", params={"limit": 501}
        )

    assert audit.status_code == 200
    body = audit.json()
    assert body["relative_path"] == "audit.pdf"
    assert body["counts"]["total"] == 2
    assert body["counts"]["native"] == 1
    assert body["counts"]["quarantine"] == 1
    assert body["pages"][0]["indexable"] is True
    assert attention.status_code == 200
    attention_body = attention.json()
    assert attention_body["total_matching"] == 1
    assert attention_body["pages"][0]["page"] == 2
    assert attention_body["pages"][0]["indexable"] is False
    assert attention_body["pages"][0]["diagnostics"].count("empty-text") == 1
    assert missing.status_code == 404
    assert invalid_limit.status_code == 422


def test_file_endpoint_only_serves_corpus_pdf(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    pdf = settings.corpus_root / "sample.pdf"
    with fitz.open() as document:
        page = document.new_page(width=320, height=240)
        page.insert_text((40, 80), "SecKnow page preview")
        document.save(pdf)
    text = settings.corpus_root / "notes.txt"
    text.write_text("not previewable", encoding="utf-8")

    with TestClient(create_app(settings)) as client:
        assert client.get("/api/files/sample.pdf").status_code == 200
        page_image = client.get("/api/pages/sample.pdf?page=1&dpi=96")
        missing_page = client.get("/api/pages/sample.pdf?page=2")
        assert client.get("/api/pages/notes.txt?page=1").status_code == 415
        assert client.get("/api/files/notes.txt").status_code == 415
        assert client.get("/api/files/../outside.pdf").status_code in {400, 404}

    assert page_image.status_code == 200
    assert page_image.headers["content-type"] == "image/png"
    assert page_image.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert missing_page.status_code == 404
