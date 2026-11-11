from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from secknow.config import RuntimeMode, Settings
from secknow.stores.metadata import JobLeaseLostError, MetadataStore
from secknow.worker import Worker


def test_worker_leases_and_completes_ingest_job(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    data = tmp_path / "data"
    corpus.mkdir()
    (corpus / "guide.txt").write_text(
        "CVE-2024-1234 是用于测试的漏洞编号。",
        encoding="utf-8",
    )
    settings = Settings(
        mode=RuntimeMode.STRICT_LEXICAL,
        corpus_root=corpus,
        data_root=data,
        database_path=data / "secknow.db",
        snapshot_root=data / "snapshots",
        model_root=tmp_path / "models",
        source_manifest=tmp_path / "sources" / "sources.yaml",
        bge_model_path=tmp_path / "models" / "bge-m3",
        reranker_model_path=tmp_path / "models" / "reranker",
        chunk_size=128,
        chunk_overlap=16,
    )
    store = MetadataStore(settings.database_path)
    store.initialize()
    store.create_job("job-1", "ingest", {"path": "guide.txt", "recursive": False})

    worker = Worker(settings, owner="test-worker")
    assert worker.run_once() is True
    job = store.get_job("job-1")
    assert job is not None
    assert job["status"] == "succeeded"
    assert job["progress"] == 1.0
    assert store.list_documents()
    assert worker.run_once() is False


def test_worker_rejects_path_escape_without_reading_it(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    data = tmp_path / "data"
    corpus.mkdir()
    settings = Settings(
        mode=RuntimeMode.STRICT_LEXICAL,
        corpus_root=corpus,
        data_root=data,
        database_path=data / "secknow.db",
        snapshot_root=data / "snapshots",
        model_root=tmp_path / "models",
        source_manifest=tmp_path / "sources" / "sources.yaml",
        bge_model_path=tmp_path / "models" / "bge-m3",
        reranker_model_path=tmp_path / "models" / "reranker",
    )
    store = MetadataStore(settings.database_path)
    store.initialize()
    store.create_job("job-bad", "ingest", {"path": "../outside.txt"})

    assert Worker(settings, owner="test-worker").run_once() is True
    job = store.get_job("job-bad")
    assert job is not None
    assert job["status"] == "failed"
    assert "relative" in job["error"]


def test_worker_continues_after_one_document_fails(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    data = tmp_path / "data"
    corpus.mkdir()
    (corpus / "bad.txt").write_text("bad", encoding="utf-8")
    (corpus / "good.txt").write_text("good", encoding="utf-8")
    settings = Settings(
        mode=RuntimeMode.STRICT_LEXICAL,
        corpus_root=corpus,
        data_root=data,
        database_path=data / "secknow.db",
        snapshot_root=data / "snapshots",
        model_root=tmp_path / "models",
        source_manifest=tmp_path / "sources" / "sources.yaml",
        bge_model_path=tmp_path / "models" / "bge-m3",
        reranker_model_path=tmp_path / "models" / "reranker",
    )
    store = MetadataStore(settings.database_path)
    store.initialize()
    store.create_job("job-partial", "ingest", {"path": ".", "recursive": True})
    worker = Worker(settings, owner="test-worker")
    attempted: list[str] = []

    class PartialPipeline:
        @staticmethod
        def ingest_path(path: Path) -> object:
            attempted.append(path.name)
            if path.name == "bad.txt":
                raise ValueError("fixture parse failure")
            return type("Result", (), {"warnings": ()})()

    worker.pipeline = PartialPipeline()  # type: ignore[assignment]
    assert worker.run_once() is True

    job = store.get_job("job-partial")
    assert job is not None
    assert attempted == ["bad.txt", "good.txt"]
    assert job["status"] == "succeeded"
    assert job["stage"] == "completed-with-1-issues"
    assert "bad.txt: ValueError: fixture parse failure" in job["error"]


def test_job_lease_renews_and_rejects_wrong_owner(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "secknow.db")
    store.initialize()
    store.create_job("job-lease", "ingest", {"path": ".", "recursive": True})

    leased = store.lease_job("worker-a", lease_seconds=2)
    assert leased is not None
    first_expiry = datetime.fromisoformat(str(leased["lease_until"]))
    store.update_job("job-lease", owner="worker-a", lease_seconds=120)

    renewed = store.get_job("job-lease")
    assert renewed is not None
    assert datetime.fromisoformat(str(renewed["lease_until"])) > first_expiry
    assert store.lease_job("worker-b") is None
    with pytest.raises(JobLeaseLostError):
        store.update_job("job-lease", progress=0.5, owner="worker-b")

    store.update_job(
        "job-lease",
        status="succeeded",
        stage="completed",
        progress=1.0,
        owner="worker-a",
    )
    completed = store.get_job("job-lease")
    assert completed is not None
    assert completed["owner"] is None
    assert completed["lease_until"] is None


def test_duplicate_jobs_are_reused_and_legacy_queue_is_superseded(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "secknow.db")
    store.initialize()
    payload = {"path": ".", "recursive": True}
    assert store.create_job("job-first", "ingest", payload) == "job-first"
    assert store.create_job("job-duplicate", "ingest", payload) == "job-first"
    assert store.get_job("job-duplicate") is None

    now = datetime.now(UTC).isoformat()
    with store.transaction() as db:
        db.execute(
            """
            INSERT INTO jobs(
                job_id, kind, status, payload_json, created_at, updated_at
            ) VALUES (?, 'ingest', 'queued', ?, ?, ?)
            """,
            ("job-legacy", '{"path": ".", "recursive": true}', now, now),
        )

    leased = store.lease_job("worker-a")
    assert leased is not None
    assert leased["job_id"] == "job-first"
    legacy = store.get_job("job-legacy")
    assert legacy is not None
    assert legacy["status"] == "superseded"
    assert legacy["stage"] == "superseded"


def test_worker_does_not_overwrite_job_after_lease_takeover(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    data = tmp_path / "data"
    corpus.mkdir()
    source = corpus / "guide.txt"
    source.write_text("ROP chain", encoding="utf-8")
    settings = Settings(
        mode=RuntimeMode.STRICT_LEXICAL,
        corpus_root=corpus,
        data_root=data,
        database_path=data / "secknow.db",
        snapshot_root=data / "snapshots",
        model_root=tmp_path / "models",
        source_manifest=tmp_path / "sources" / "sources.yaml",
        bge_model_path=tmp_path / "models" / "bge-m3",
        reranker_model_path=tmp_path / "models" / "reranker",
    )
    store = MetadataStore(settings.database_path)
    store.initialize()
    store.create_job("job-takeover", "ingest", {"path": "guide.txt", "recursive": False})
    worker = Worker(settings, owner="worker-old")

    class TakeoverPipeline:
        @staticmethod
        def ingest_path(_path: Path) -> object:
            with store.transaction() as db:
                db.execute(
                    "UPDATE jobs SET owner='worker-new' WHERE job_id='job-takeover'"
                )
            return type("Result", (), {"warnings": ()})()

    worker.pipeline = TakeoverPipeline()  # type: ignore[assignment]
    assert worker.run_once() is True

    job = store.get_job("job-takeover")
    assert job is not None
    assert job["status"] == "running"
    assert job["owner"] == "worker-new"
