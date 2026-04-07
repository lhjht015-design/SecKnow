from __future__ import annotations

"""Phase 1 联调脚本：把 data/ 文件批量走 4.1 流水线并写入 4.3。"""

import argparse
import os
from collections import Counter
from pathlib import Path

from secknow.text_processing import DocumentTextPipeline
from secknow.vector_store.factory import build_vector_service
from secknow.vector_store.models import ChunkRecord, RecordType, ZoneId


DEFAULT_EXTS = {".txt", ".md", ".docx"}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Phase 1: 4.1 -> 4.3 联调脚本")
    parser.add_argument(
        "--data-dir",
        default="../../data",
        help="测试数据目录（默认指向仓库根下 data 目录）",
    )
    parser.add_argument(
        "--zone-id",
        default="cyber",
        choices=["cyber", "ai", "crypto"],
        help="写入分区",
    )
    parser.add_argument(
        "--record-type",
        default="knowledge",
        choices=["knowledge", "baseline"],
        help="记录类型",
    )
    parser.add_argument(
        "--mode",
        default="offline",
        choices=["offline", "online"],
        help="4.3 存储模式：offline(FAISS+SQLite) 或 online(Qdrant)",
    )
    parser.add_argument(
        "--embedding-mode",
        default=os.getenv("EMBEDDING_MODE", "sbert"),
        choices=["sbert", "fake"],
        help="编码模式",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        help="SBERT 模型名",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=int(os.getenv("EMBEDDING_DIM", "384")),
        help="向量维度（默认 384）",
    )
    parser.add_argument(
        "--chunk-strategy",
        default="hybrid",
        choices=["hybrid", "fixed_window"],
        help="分块策略",
    )
    parser.add_argument("--max-tokens", type=int, default=300, help="分块窗口大小")
    parser.add_argument("--overlap", type=int, default=50, help="分块 overlap")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多处理文件数，0 表示不限制",
    )
    parser.add_argument(
        "--exts",
        default=",".join(sorted(DEFAULT_EXTS)),
        help="文件扩展名白名单，逗号分隔（如 .txt,.md,.docx）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只跑 4.1，不写入 4.3",
    )

    # offline 参数
    parser.add_argument("--db-path", default="db/secknow.sqlite3", help="离线 SQLite 路径")
    parser.add_argument("--index-dir", default="db/faiss", help="离线 FAISS 索引目录")

    # online 参数
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant 主机")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant 端口")
    return parser.parse_args()


def list_files(data_dir: Path, allow_exts: set[str], limit: int = 0) -> list[Path]:
    """列出待处理文件。"""
    files = [
        path
        for path in sorted(data_dir.iterdir())
        if path.is_file() and path.suffix.lower() in allow_exts
    ]
    if limit > 0:
        return files[:limit]
    return files


def build_service(args: argparse.Namespace):
    """构建 4.3 服务。"""
    if args.mode == "offline":
        return build_vector_service(
            mode="offline",
            db_path=args.db_path,
            index_dir=args.index_dir,
            embedding_dim=args.embedding_dim,
            embedding_model=args.embedding_model,
        )
    return build_vector_service(
        mode="online",
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        embedding_model=args.embedding_model,
    )


def process_all(
    *,
    pipeline: DocumentTextPipeline,
    files: list[Path],
    zone_id: ZoneId,
    record_type: RecordType,
) -> tuple[list[ChunkRecord], list[tuple[Path, str]]]:
    """批量执行 4.1 `process_file`，返回 records 与失败列表。"""
    all_records: list[ChunkRecord] = []
    errors: list[tuple[Path, str]] = []

    for idx, file_path in enumerate(files, start=1):
        try:
            records = pipeline.process_file(
                file_path=file_path,
                zone_id=zone_id,
                record_type=record_type,
            )
            all_records.extend(records)
            print(f"[{idx}/{len(files)}] OK   {file_path.name} -> records={len(records)}")
        except Exception as exc:  # noqa: BLE001 - 联调脚本需要持续处理后续文件
            errors.append((file_path, str(exc)))
            print(f"[{idx}/{len(files)}] FAIL {file_path.name} -> {exc}")

    return all_records, errors


def main() -> None:
    """脚本主流程。"""
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    # 默认从 backend/scripts 回到仓库根，再定位 data 目录。
    data_dir = (script_dir / args.data_dir).resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir 不存在或不是目录: {data_dir}")

    allow_exts = {item.strip().lower() for item in args.exts.split(",") if item.strip()}
    files = list_files(data_dir=data_dir, allow_exts=allow_exts, limit=args.limit)
    if not files:
        print(f"未匹配到文件，allow_exts={sorted(allow_exts)} data_dir={data_dir}")
        return

    pipeline = DocumentTextPipeline(
        chunk_strategy=args.chunk_strategy,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
        embedding_mode=args.embedding_mode,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
    )

    zone_id: ZoneId = args.zone_id
    record_type: RecordType = args.record_type
    records, errors = process_all(
        pipeline=pipeline,
        files=files,
        zone_id=zone_id,
        record_type=record_type,
    )

    print("\n=== 4.1 汇总 ===")
    print(f"data_dir={data_dir}")
    print(f"files_total={len(files)} files_failed={len(errors)}")
    print(f"records_total={len(records)}")

    ext_counter = Counter(path.suffix.lower() for path in files)
    print(f"ext_distribution={dict(ext_counter)}")

    if errors:
        print("\n失败文件：")
        for path, msg in errors:
            print(f"- {path.name}: {msg}")

    if args.dry_run:
        print("\n已启用 --dry-run，仅验证 4.1 产出，不执行 upsert。")
        return

    if not records:
        print("\n没有可写入记录，跳过 upsert。")
        return

    service = build_service(args)
    result = service.upsert(zone_id=zone_id, records=records)

    print("\n=== 4.3 upsert 结果 ===")
    print(f"zone_id={result.zone_id}")
    print(f"attempted={result.attempted} inserted={result.inserted}")
    print(f"chunk_id_count={len(result.chunk_ids)}")


if __name__ == "__main__":
    main()
