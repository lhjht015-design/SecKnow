from __future__ import annotations

from pathlib import Path

from secknow.text_processing.pipeline.run_pipeline import DocumentTextPipeline, run_pipeline


def test_pipeline_produces_records(tmp_path: Path, monkeypatch) -> None:
    """process_file 应返回可直接 upsert 的 ChunkRecord 列表。"""
    monkeypatch.setenv("EMBEDDING_MODE", "fake")
    monkeypatch.setenv("EMBEDDING_DIM", "384")

    file_path = tmp_path / "knowledge.md"
    file_path.write_text(
        "# 应急响应\n发生安全事件时应先隔离主机并保留日志。\n发生安全事件时应先隔离主机并保留日志。",
        encoding="utf-8",
    )

    pipeline = DocumentTextPipeline(max_tokens=30, overlap=5)
    records = pipeline.process_file(file_path, zone_id="cyber")

    assert len(records) >= 1
    assert all(len(record.vector) == 384 for record in records)
    assert all(record.metadata.zone_id == "cyber" for record in records)
    assert all(record.metadata.record_type == "knowledge" for record in records)


def test_run_pipeline_functional_entry(tmp_path: Path, monkeypatch) -> None:
    """run_pipeline 函数式入口应可直接返回 records。"""
    monkeypatch.setenv("EMBEDDING_MODE", "fake")
    file_path = tmp_path / "mini.txt"
    file_path.write_text("最小闭环验证文本", encoding="utf-8")

    records = run_pipeline(file_path, zone_id="ai")
    assert len(records) == 1
    assert records[0].metadata.zone_id == "ai"


def test_process_text_no_disk_write(tmp_path: Path, monkeypatch) -> None:
    """process_text 不应要求磁盘上已存在源文件。"""
    monkeypatch.setenv("EMBEDDING_MODE", "fake")
    monkeypatch.setenv("EMBEDDING_DIM", "384")

    pipeline = DocumentTextPipeline(max_tokens=40, overlap=5)
    target = tmp_path / "virtual" / "note.md"
    records = pipeline.process_text(
        "# 标题\n正文一行。",
        source_path=str(target),
        zone_id="crypto",
    )
    assert len(records) >= 1
    assert not target.exists()
