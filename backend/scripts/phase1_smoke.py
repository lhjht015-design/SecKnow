from __future__ import annotations

import hashlib
import os
import time

from secknow.vector_store.factory import build_offline_service, build_online_service
from secknow.vector_store.models import ChunkMetadata, ChunkRecord, RecordType, ZoneId


def fake_vec(seed: str, dim: int = 384) -> list[float]:
    """根据种子文本生成确定性的伪向量，便于本地重复验证。"""
    # 这里只是为了 smoke test，目标不是“语义正确”，而是“同一输入每次都能得到同一向量”。
    # 这样我们在没有真实 embedding 模型参与时，也能稳定验证：
    # 1. upsert 是否成功
    # 2. search 是否能返回结果
    # 3. baseline 是否能被单独提取
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    base = [(digest[i % len(digest)] / 255.0) for i in range(dim)]
    norm = sum(v * v for v in base) ** 0.5
    return [v / norm for v in base] if norm else base


def build_record(
    *,
    doc_id: str,
    zone_id: ZoneId,
    filename: str,
    text: str,
    chunk_index: int,
    chunk_count: int,
    record_type: RecordType = "knowledge",
    language: str = "zh",
) -> ChunkRecord:
    """构造一条标准 ChunkRecord，减少 smoke 脚本里的样板代码。"""
    now = int(time.time())
    return ChunkRecord(
        text=text,
        vector=fake_vec(f"{doc_id}:{chunk_index}:{record_type}:{text}"),
        metadata=ChunkMetadata(
            doc_id=doc_id,
            zone_id=zone_id,
            filename=filename,
            source_path=f"/tmp/{filename}",
            extension=".md",
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            char_len=len(text),
            content_hash=hashlib.sha1(text.encode("utf-8")).hexdigest(),
            mtime=now,
            size_bytes=len(text.encode("utf-8")),
            file_type="markdown",
            language=language,
            record_type=record_type,
        ),
    )


def build_demo_records() -> list[ChunkRecord]:
    """伪造一组最小但完整的演示数据。"""
    # 这里刻意放了两类数据：
    # 1. knowledge：给普通检索链路用
    # 2. baseline：给 4.2 安全审查链路用
    #
    # 这样一次 smoke 就能验证：
    # - 普通 search 默认只搜 knowledge
    # - get_baseline 只取 baseline
    return [
        build_record(
            doc_id="kb-cyber-001",
            zone_id="cyber",
            filename="incident_response.md",
            text="发生主机入侵事件后，优先隔离主机、保留日志并启动应急响应流程。",
            chunk_index=0,
            chunk_count=3,
        ),
        build_record(
            doc_id="kb-cyber-001",
            zone_id="cyber",
            filename="incident_response.md",
            text="日志审计应重点关注异常登录、提权行为、横向移动和敏感文件访问。",
            chunk_index=1,
            chunk_count=3,
        ),
        build_record(
            doc_id="kb-cyber-002",
            zone_id="cyber",
            filename="asset_security.md",
            text="重要资产应建立分级台账，并落实最小权限、口令轮换和补丁治理机制。",
            chunk_index=0,
            chunk_count=1,
        ),
        build_record(
            doc_id="baseline-cyber-001",
            zone_id="cyber",
            filename="baseline_windows.md",
            text="Windows 主机基线要求：关闭高危端口、启用主机防火墙、限制管理员远程登录。",
            chunk_index=0,
            chunk_count=2,
            record_type="baseline",
        ),
        build_record(
            doc_id="baseline-cyber-001",
            zone_id="cyber",
            filename="baseline_windows.md",
            text="安全基线要求：关键日志保留不少于六个月，并开启异常告警与集中审计。",
            chunk_index=1,
            chunk_count=2,
            record_type="baseline",
        ),
    ]


def build_service():
    """根据环境变量决定连接在线 Qdrant 还是离线 FAISS+SQLite。"""
    # 默认走 online，方便你直接验证本地 Docker Qdrant。
    # 如果想切回离线模式，可以：
    # VECTOR_MODE=offline PYTHONPATH=src python -m scripts.phase1_smoke
    mode = os.getenv("VECTOR_MODE", "online").lower()
    if mode == "offline":
        return build_offline_service()

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    return build_online_service(qdrant_host=qdrant_host, qdrant_port=qdrant_port)


def main() -> None:
    """执行第一阶段最小冒烟验证。"""
    # 整个 smoke 的目标不是测性能，而是快速确认第一阶段的几条关键链路都通：
    # 1. 服务能正常初始化
    # 2. knowledge / baseline 能成功 upsert
    # 3. 普通向量检索能返回 knowledge
    # 4. baseline 能通过 get_baseline 单独取出
    # 5. hybrid_search 能跑通 dense + sparse 融合
    service = build_service()
    records = build_demo_records()

    print("=== Phase 1 Smoke Start ===")
    print(f"record_count={len(records)}")

    upsert_result = service.upsert(zone_id="cyber", records=records)
    print(
        f"upsert attempted={upsert_result.attempted} inserted={upsert_result.inserted}"
    )

    # 这里用第一条 knowledge 的向量作为查询向量，方便观察最相近结果是否能返回。
    query_record = next(
        record for record in records if record.metadata.record_type == "knowledge"
    )
    dense_hits = service.search(zone_id="cyber", query_vec=query_record.vector, top_k=3)
    print(f"dense_hits={len(dense_hits)}")
    for idx, hit in enumerate(dense_hits, start=1):
        print(
            f"dense[{idx}] score={hit.score:.6f} "
            f"record_type={hit.metadata.get('record_type')} filename={hit.metadata.get('filename')}"
        )

    # hybrid_search 会同时走：
    # - dense_store.search()
    # - sparse_index.search()
    # 然后在兼容层做 RRF 融合。
    hybrid_hits = service.hybrid_search(
        zone_id="cyber",
        query="日志审计 异常登录 安全基线",
        query_vec=query_record.vector,
        top_k=3,
    )
    print(f"hybrid_hits={len(hybrid_hits)}")
    for idx, hit in enumerate(hybrid_hits, start=1):
        print(
            f"hybrid[{idx}] score={hit.score:.6f} "
            f"record_type={hit.metadata.get('record_type')} filename={hit.metadata.get('filename')}"
        )

    baseline = service.get_baseline(zone_id="cyber")
    print(f"baseline_count={baseline.count}")
    for idx, metadata in enumerate(baseline.metadatas[:3], start=1):
        print(
            f"baseline[{idx}] chunk_id={metadata.get('chunk_id')} "
            f"filename={metadata.get('filename')} record_type={metadata.get('record_type')}"
        )

    print("=== Phase 1 Smoke Done ===")


if __name__ == "__main__":
    main()
