from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from secknow.config import RuntimeMode, get_settings
from secknow.evaluation import mean_metrics, retrieval_metrics
from secknow.ingestion.discovery import DocumentDiscovery
from secknow.ingestion.pipeline import IngestionPipeline, IngestionResult
from secknow.main import run as run_api
from secknow.retrieval.service import RetrievalService
from secknow.runtime import capability_report
from secknow.stores.metadata import MetadataStore
from secknow.worker import run as run_worker

# Windows 控制台默认 GBK。检索结果里经常出现 ✓/μ/特殊符号，
# 直接 print 会抛 UnicodeEncodeError 把整条命令打断。统一改成 UTF-8 + 可替换。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="secknow")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="initialize data directories and metadata")
    commands.add_parser("discover", help="list supported corpus documents")
    commands.add_parser("doctor", help="report effective local runtime capabilities")
    ingest = commands.add_parser("ingest", help="index one corpus-relative path or all files")
    ingest.add_argument("path", nargs="?", default=".")
    search = commands.add_parser("search", help="run a knowledge search")
    search.add_argument("query")
    search.add_argument("--top-k", type=int)
    search.add_argument("--mode", choices=[mode.value for mode in RuntimeMode])
    evaluate = commands.add_parser("evaluate", help="evaluate retrieval against JSONL judgements")
    evaluate.add_argument("gold", type=Path)
    evaluate.add_argument("--top-k", type=int, default=10, choices=range(1, 51))
    evaluate.add_argument("--mode", choices=[mode.value for mode in RuntimeMode])
    commands.add_parser("serve", help="start the local web API")
    worker = commands.add_parser("worker", help="start the persistent index worker")
    worker.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    settings = get_settings()
    if args.command == "init":
        settings.ensure_directories()
        MetadataStore(settings.database_path).initialize()
        print(f"initialized {settings.data_root}")
        return
    if args.command == "discover":
        for document in DocumentDiscovery(settings.corpus_root).discover():
            print(json.dumps(document.model_dump(mode="json"), ensure_ascii=False))
        return
    if args.command == "doctor":
        store = MetadataStore(settings.database_path)
        store.initialize()
        report = capability_report(
            settings,
            tfidf_ready=bool(store.get_setting("active_tfidf_version")),
            graph_ready=bool(store.get_setting("active_graph_version")),
        )
        print(json.dumps({"mode": settings.mode.value, **report}, ensure_ascii=False, indent=2))
        return
    if args.command == "ingest":
        pipeline = IngestionPipeline(settings)
        if args.path == ".":
            results = pipeline.ingest_all()
        else:
            target = _corpus_path(settings.corpus_root, args.path)
            results = (pipeline.ingest_path(target),)
        print(json.dumps([_result_summary(result) for result in results], ensure_ascii=False))
        return
    if args.command == "search":
        selected = settings
        if args.mode:
            selected = settings.model_copy(update={"mode": RuntimeMode(args.mode)})
        response = RetrievalService(selected).search(args.query, args.top_k)
        print(response.model_dump_json(indent=2))
        return
    if args.command == "evaluate":
        selected = settings
        if args.mode:
            selected = settings.model_copy(update={"mode": RuntimeMode(args.mode)})
        service = RetrievalService(selected)
        metric_rows = []
        with args.gold.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                query = str(row.get("query", "")).strip()
                relevant = {str(item) for item in row.get("relevant_chunk_ids", []) if item}
                if not query or not relevant:
                    raise SystemExit(f"invalid judgement at line {line_number}")
                response = service.search(query, args.top_k)
                metric_rows.append(
                    retrieval_metrics(
                        relevant,
                        [hit.chunk.chunk_id for hit in response.hits],
                        args.top_k,
                    )
                )
        metrics = mean_metrics(metric_rows)
        print(
            json.dumps(
                {"queries": len(metric_rows), "k": args.top_k, **metrics.__dict__},
                indent=2,
            )
        )
        return
    if args.command == "serve":
        run_api()
        return
    run_worker(["--once"] if args.once else [])


def _corpus_path(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit("path must be relative to the corpus root")
    target = (root.resolve() / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise SystemExit("path escapes the corpus root") from error
    if not target.is_file():
        raise SystemExit(f"file does not exist: {relative_path}")
    return target


def _result_summary(result: IngestionResult) -> dict[str, object]:
    return {
        "source": result.document.relative_path,
        "version": result.document.version_id,
        "chunks": len(result.chunks),
        "warnings": list(result.warnings),
    }


if __name__ == "__main__":
    main()
