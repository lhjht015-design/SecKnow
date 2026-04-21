from __future__ import annotations

"""4.1 -> 4.3 单文件入库 CLI。"""

import argparse
import json
from pathlib import Path

from secknow.text_processing import DocumentTextPipeline
from secknow.text_processing.schemas.pipeline_result import PipelineResult
from secknow.vector_store.factory import build_vector_service


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="将单个文件经过 4.1 处理后写入 4.3，适合本地联调。"
    )
    parser.add_argument("file", help="待入库文件路径")
    parser.add_argument(
        "--zone-id",
        default="cyber",
        choices=["cyber", "ai", "crypto"],
        help="目标知识分区，默认 cyber",
    )
    parser.add_argument(
        "--record-type",
        default="knowledge",
        choices=["knowledge", "baseline"],
        help="记录类型，默认 knowledge",
    )
    parser.add_argument(
        "--mode",
        default="online",
        choices=["online", "offline"],
        help="向量存储模式，默认 online",
    )
    parser.add_argument(
        "--embedding-mode",
        default="sbert",
        choices=["sbert", "fake"],
        help="编码模式，默认 sbert",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="编码模型名，默认与 4.1 / 4.3 对齐",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=384,
        help="向量维度，默认 384",
    )
    parser.add_argument(
        "--chunk-strategy",
        default="hybrid",
        choices=["hybrid", "fixed_window", "paragraph", "line", "semantic"],
        help="分块策略，默认 hybrid",
    )
    parser.add_argument(
        "--dedup-strategy",
        default="exact",
        choices=["exact", "minhash"],
        help="去重策略，默认 exact",
    )
    parser.add_argument("--max-tokens", type=int, default=300, help="分块窗口大小")
    parser.add_argument("--overlap", type=int, default=50, help="分块 overlap")
    parser.add_argument(
        "--language",
        default=None,
        help="可选语言标记，例如 zh / en / python",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只跑 4.1，不执行写库",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出摘要信息",
    )

    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant 主机")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant 端口")
    parser.add_argument(
        "--sparse-db-path",
        default="db/online_sparse.sqlite3",
        help="online 模式下用于持久化 BM25/FTS 的本地 SQLite 路径",
    )

    parser.add_argument("--db-path", default="db/secknow.sqlite3", help="offline SQLite 路径")
    parser.add_argument("--index-dir", default="db/faiss", help="offline FAISS 索引目录")
    return parser.parse_args()


def build_service(args: argparse.Namespace):
    """按运行模式构建 4.3 服务。"""
    if args.mode == "online":
        return build_vector_service(
            mode="online",
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            embedding_model=args.embedding_model,
            sparse_db_path=args.sparse_db_path,
        )
    return build_vector_service(
        mode="offline",
        db_path=args.db_path,
        index_dir=args.index_dir,
        embedding_dim=args.embedding_dim,
        embedding_model=args.embedding_model,
    )


def print_pipeline_log(result: PipelineResult) -> None:
    """打印本次 4.1 流水线的关键信息。"""
    doc = result.loaded_document
    print("=== 4.1 流水线日志 ===")
    print(f"source_path={doc.source_path}")
    print(f"filename={doc.filename}")
    print(f"file_type={doc.file_type} extension={doc.extension}")
    print(f"doc_id={doc.doc_id}")
    print(f"raw_text_len={len(result.raw_document.text)}")
    print(f"chunks_before_dedup={len(result.chunks)}")
    print(f"chunks_after_dedup={len(result.deduped_chunks)}")
    print(f"dropped_duplicates={result.dropped_duplicates}")
    print(f"records_ready={len(result.records)}")

    for chunk in result.deduped_chunks:
        preview = chunk.text.replace("\n", " ").strip()
        if len(preview) > 80:
            preview = f"{preview[:80]}..."
        print(
            f"chunk[{chunk.chunk_index + 1}/{chunk.chunk_count}] "
            f"len={len(chunk.text)} preview={preview}"
        )


def print_json_summary(args: argparse.Namespace, result: PipelineResult, inserted: int) -> None:
    """输出 JSON 摘要。"""
    payload = {
        "file": args.file,
        "zone_id": args.zone_id,
        "record_type": args.record_type,
        "mode": args.mode,
        "doc_id": result.loaded_document.doc_id,
        "file_type": result.loaded_document.file_type,
        "chunks_before_dedup": len(result.chunks),
        "chunks_after_dedup": len(result.deduped_chunks),
        "dropped_duplicates": result.dropped_duplicates,
        "records_ready": len(result.records),
        "inserted": inserted,
        "chunk_ids": [record.metadata.chunk_id for record in result.records],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    """CLI 主流程。"""
    args = parse_args()
    file_path = Path(args.file).resolve()
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"文件不存在或不是普通文件: {file_path}")

    pipeline = DocumentTextPipeline(
        chunk_strategy=args.chunk_strategy,
        dedup_strategy=args.dedup_strategy,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
        embedding_mode=args.embedding_mode,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
    )
    result = pipeline.process_file(
        file_path=file_path,
        zone_id=args.zone_id,
        record_type=args.record_type,
        language=args.language,
        return_result=True,
    )
    assert isinstance(result, PipelineResult)

    print_pipeline_log(result)

    inserted = 0
    if args.dry_run:
        print("\n已启用 --dry-run，本次不会执行写库。")
    elif not result.records:
        print("\n没有可写入的记录，跳过入库。")
    else:
        service = build_service(args)
        upsert_result = service.upsert(zone_id=args.zone_id, records=result.records)
        inserted = upsert_result.inserted
        print("\n=== 4.3 写库结果 ===")
        print(f"mode={args.mode}")
        print(f"zone_id={upsert_result.zone_id}")
        print(f"attempted={upsert_result.attempted}")
        print(f"inserted={upsert_result.inserted}")
        print(f"chunk_id_count={len(upsert_result.chunk_ids)}")

    if args.json:
        print()
        print_json_summary(args, result, inserted)


if __name__ == "__main__":
    main()
