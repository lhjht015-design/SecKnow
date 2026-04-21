from __future__ import annotations

"""4.4 推理模块联调 CLI。"""

import argparse
import json

from secknow.inference import InferenceService
from secknow.vector_store.factory import build_vector_service


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。"""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--mode",
        default="online",
        choices=["online", "offline"],
        help="向量存储模式，默认 online",
    )
    common.add_argument("--qdrant-host", default="localhost", help="Qdrant 主机")
    common.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant 端口")
    common.add_argument(
        "--sparse-db-path",
        default="db/online_sparse.sqlite3",
        help="online 模式下用于持久化 BM25/FTS 的本地 SQLite 路径",
    )
    common.add_argument("--db-path", default="db/secknow.sqlite3", help="offline SQLite 路径")
    common.add_argument("--index-dir", default="db/faiss", help="offline FAISS 索引目录")

    parser = argparse.ArgumentParser(
        description="调用 4.4 推理入口，适合本地联调 4.3 / 4.4。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    semantic = subparsers.add_parser(
        "semantic-search",
        help="执行语义检索",
        parents=[common],
    )
    semantic.add_argument("--query", required=True, help="查询文本")
    semantic.add_argument(
        "--zone-id",
        default="cyber",
        choices=["cyber", "ai", "crypto"],
        help="目标知识分区，默认 cyber",
    )
    semantic.add_argument("--top-k", type=int, default=5, help="返回结果数量，默认 5")
    semantic.add_argument(
        "--record-type",
        default="knowledge",
        choices=["knowledge", "baseline"],
        help="检索记录类型，默认 knowledge",
    )
    semantic.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出结果",
    )

    scan = subparsers.add_parser(
        "code-risk-scan",
        help="执行代码风险扫描",
        parents=[common],
    )
    source_group = scan.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--code", help="直接传入待扫描代码")
    source_group.add_argument("--code-file", help="从文件读取待扫描代码")
    scan.add_argument("--lang", required=True, help="代码语言，例如 python")
    scan.add_argument(
        "--zone-id",
        default="cyber",
        choices=["cyber", "ai", "crypto"],
        help="目标知识分区，默认 cyber",
    )
    scan.add_argument("--top-k", type=int, default=5, help="每个扫描单元参考结果数量，默认 5")
    scan.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出结果",
    )

    return parser


def build_service(args: argparse.Namespace):
    """按运行模式构建 4.3 服务。"""
    if args.mode == "online":
        vector_service = build_vector_service(
            mode="online",
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            sparse_db_path=args.sparse_db_path,
        )
    else:
        vector_service = build_vector_service(
            mode="offline",
            db_path=args.db_path,
            index_dir=args.index_dir,
        )
    return InferenceService(vector_service=vector_service)


def print_semantic_results(args: argparse.Namespace, results) -> None:
    """打印语义检索结果。"""
    if args.json:
        print(
            json.dumps(
                [result.model_dump() for result in results],
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print("=== 4.4 语义检索结果 ===")
    print(f"query={args.query}")
    print(f"zone_id={args.zone_id}")
    print(f"top_k={args.top_k}")
    print(f"result_count={len(results)}")

    for idx, result in enumerate(results, start=1):
        print(f"\n[{idx}] score={result.score:.6f}")
        print(f"doc_id={result.doc_id}")
        print(f"filename={result.filename}")
        print(f"source_path={result.source_path}")
        print(f"record_type={result.record_type}")
        print(f"text={result.text}")


def print_code_scan_results(args: argparse.Namespace, result) -> None:
    """打印代码风险扫描结果。"""
    if args.json:
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
        return

    print("=== 4.4 代码风险扫描结果 ===")
    print(f"lang={result.lang}")
    print(f"zone_id={result.zone_id}")
    print(f"item_count={len(result.items)}")

    for idx, item in enumerate(result.items, start=1):
        print(f"\n[{idx}] severity={item.severity}")
        print(f"summary={item.summary}")
        print("code_unit:")
        print(item.code_unit)
        print(f"reference_count={len(item.references)}")
        for ref_idx, ref in enumerate(item.references, start=1):
            print(
                f"  - ref[{ref_idx}] score={ref.score:.6f} "
                f"doc_id={ref.doc_id} filename={ref.filename}"
            )


def load_code_input(args: argparse.Namespace) -> str:
    """读取代码扫描输入。"""
    if args.code is not None:
        return args.code
    with open(args.code_file, "r", encoding="utf-8") as handle:
        return handle.read()


def main() -> None:
    """CLI 主流程。"""
    parser = build_parser()
    args = parser.parse_args()
    service = build_service(args)

    if args.command == "semantic-search":
        results = service.semantic_search(
            query=args.query,
            zone_id=args.zone_id,
            top_k=args.top_k,
            filters={"record_type": args.record_type},
        )
        print_semantic_results(args, results)
        return

    if args.command == "code-risk-scan":
        code_snippet = load_code_input(args)
        result = service.code_risk_scan(
            code_snippet=code_snippet,
            lang=args.lang,
            zone_id=args.zone_id,
            top_k=args.top_k,
        )
        print_code_scan_results(args, result)
        return

    raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
