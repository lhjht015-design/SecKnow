from __future__ import annotations

from secknow.text_processing.dedupers.router import dedup_chunks


def test_exact_dedup() -> None:
    """exact 去重应保留首次出现的 chunk。"""
    chunks = ["A", "B", "A", "C", "B"]
    assert dedup_chunks(chunks, strategy="exact") == ["A", "B", "C"]


def test_minhash_dedup_near_duplicate() -> None:
    """minhash 应合并高度相似的文本块。"""
    a = "应急响应流程包含隔离主机与保留日志两个关键步骤"
    b = "应急响应流程包含隔离主机与保留日志两个关键步骤。"
    c = "完全不同的内容讨论数据库索引优化"
    out = dedup_chunks([a, b, c], strategy="minhash")
    assert len(out) == 2
    assert c in out
    assert a in out or b in out
