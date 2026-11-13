"""验证链推荐：入口识别 → 候选链 → 知识证据。产品内不执行目标程序。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from secknow.solving import binary, chains, entries
from secknow.stores.metadata import MetadataStore

_TECHNIQUE_QUERIES = {
    "unsafe_copy": "栈溢出 ret2win gets payload 偏移",
    "command_injection": "命令注入 system payload",
    "format_string": "格式化字符串 泄漏 覆写",
    "blackbox_check": "黑盒 变换 反解 探测",
}


def recommend(
    store: MetadataStore,
    *,
    source: str | None = None,
    filename: str | None = None,
    binary_path: Path | None = None,
    max_evidence: int = 4,
) -> dict[str, Any]:
    warnings: list[str] = []
    profile: binary.BinaryProfile | None = None
    if binary_path is not None:
        if not binary.dependencies_available():
            warnings.append("未安装 pyelftools/capstone，已跳过二进制分析")
        else:
            try:
                profile = binary.analyze(binary_path)
            except (OSError, ValueError, RuntimeError) as error:
                warnings.append(f"二进制分析失败：{type(error).__name__}")

    detected: list[dict[str, Any]] = []
    categories: list[str] = []
    for entry in entries.detect_source_entries(source or "") if source else []:
        detected.append({**asdict(entry)})
        categories.append(entry.category)

    if profile is not None:
        for finding in profile.findings():
            if finding.category in categories:
                continue
            categories.append(finding.category)
            detected.append({
                "category": finding.category,
                "technique": finding.technique,
                "line": None,
                "text": finding.evidence,
                "confidence": "high" if finding.category != "format_string" else "medium",
            })

    chain_list = chains.build_chains(categories, profile)
    identifiers = entries.extract_identifiers(source or "") if source else []

    payload_chains = []
    for chain in chain_list:
        evidence = _evidence(store, chain, identifiers, filename, max_evidence)
        payload_chains.append({
            "chain_id": chain.chain_id,
            "title": chain.title,
            "category": chain.category,
            "status": chain.status,
            "reason": chain.reason,
            "steps": [asdict(step) for step in chain.steps],
            "evidence": evidence,
        })

    payload_binary = None
    if profile is not None:
        payload_binary = {
            "is_pie": profile.is_pie,
            "has_canary": profile.has_canary,
            "has_nx": profile.has_nx,
            "relro": profile.relro,
            "imports": len(profile.imports),
            "win_candidates": profile.win_candidates[:5],
            "stack_hints": profile.stack_hints,
            "call_graph": profile.call_graph,
        }

    if not detected:
        warnings.append("未识别到入口：未命中已知的 sink 形态")
    return {
        "entries": detected,
        "chains": payload_chains,
        "binary": payload_binary,
        "warnings": warnings,
    }


def _evidence(
    store: MetadataStore,
    chain: chains.Chain,
    identifiers: list[str],
    filename: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    terms = [_TECHNIQUE_QUERIES.get(chain.category, chain.title)]
    if identifiers:
        terms.append(" ".join(identifiers[:5]))
    if filename:
        terms.append(filename)
    query = " ".join(terms)
    try:
        hits = store.search_fts(query, limit * 2)
    except Exception:  # noqa: BLE001 - 证据检索失败不应让整个分析失败
        return []
    chunk_map = store.get_chunks([chunk_id for chunk_id, _ in hits])
    evidence: list[dict[str, Any]] = []
    for chunk_id, score in hits:
        chunk = chunk_map.get(chunk_id)
        if chunk is None:
            continue
        text = chunk.text.strip()
        if not text:
            continue
        evidence.append({
            "chunk_id": chunk_id,
            "score": round(float(score), 4),
            "title": chunk.citation.title,
            "relative_path": chunk.citation.relative_path,
            "page_start": chunk.citation.page_start,
            "section_path": list(chunk.section_path),
            "text": text[:400],
        })
        if len(evidence) >= limit:
            break
    return evidence
