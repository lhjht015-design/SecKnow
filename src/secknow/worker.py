from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Any

from secknow.api.security import indexable_files
from secknow.config import Settings, get_settings
from secknow.ingestion.pipeline import IngestionPipeline
from secknow.stores.metadata import JobLeaseLostError, MetadataStore

LOGGER = logging.getLogger("secknow.worker")


class Worker:
    def __init__(self, settings: Settings, *, owner: str | None = None) -> None:
        self.settings = settings
        self.store = MetadataStore(settings.database_path)
        self.store.initialize()
        self.pipeline = IngestionPipeline(settings, self.store)
        self.owner = owner or f"{socket.gethostname()}:{os.getpid()}"
        self.stopping = False

    def request_stop(self) -> None:
        self.stopping = True

    def run_once(self) -> bool:
        row = self.store.lease_job(self.owner)
        if row is None:
            return False
        job_id = str(row["job_id"])
        heartbeat_stop = threading.Event()
        lease_lost = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat,
            args=(job_id, heartbeat_stop, lease_lost),
            daemon=True,
        )
        heartbeat.start()
        try:
            payload = json.loads(str(row["payload_json"]))
            self._dispatch(job_id, str(row["kind"]), payload, lease_lost)
        except JobLeaseLostError:
            lease_lost.set()
            LOGGER.warning("Job %s lease was lost", job_id)
        except Exception as error:
            LOGGER.exception("Job %s failed", job_id)
            try:
                self.store.update_job(
                    job_id,
                    status="failed",
                    stage="failed",
                    progress=1.0,
                    error=f"{type(error).__name__}: {error}"[:4000],
                    owner=self.owner,
                )
            except JobLeaseLostError:
                LOGGER.warning("Job %s lease was lost while reporting failure", job_id)
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=1.0)
        return True

    def _heartbeat(
        self,
        job_id: str,
        stopping: threading.Event,
        lease_lost: threading.Event,
    ) -> None:
        while not stopping.wait(30.0):
            try:
                self.store.update_job(job_id, owner=self.owner)
            except JobLeaseLostError:
                lease_lost.set()
                return

    def run_forever(self, poll_seconds: float = 1.0) -> None:
        while not self.stopping:
            if not self.run_once():
                time.sleep(max(0.05, poll_seconds))

    def _dispatch(
        self,
        job_id: str,
        kind: str,
        payload: dict[str, Any],
        lease_lost: threading.Event,
    ) -> None:
        if kind != "ingest":
            raise ValueError(f"Unsupported job kind: {kind}")
        recursive = bool(payload.get("recursive", True))
        raw_paths = payload.get("paths")
        if isinstance(raw_paths, list):
            targets = [self._resolve_path(str(path)) for path in raw_paths]
        else:
            targets = [self._resolve_path(str(payload.get("path", ".")))]
        discovered = [path for target in targets for path in indexable_files(target, recursive)]
        files = list(dict.fromkeys(discovered))
        if not files:
            raise ValueError("No supported documents found")
        warnings: list[str] = []
        failures: list[str] = []
        succeeded = 0
        total = len(files)
        for index, path in enumerate(files, 1):
            if lease_lost.is_set():
                raise JobLeaseLostError(f"Job lease lost: {job_id}")
            if self.stopping:
                self.store.update_job(
                    job_id,
                    status="queued",
                    stage="interrupted",
                    progress=(index - 1) / total,
                    owner=self.owner,
                )
                return
            self.store.update_job(
                job_id,
                stage=f"ingesting:{path.name}",
                progress=(index - 1) / total,
                owner=self.owner,
            )
            try:
                result = self.pipeline.ingest_path(path)
            except (JobLeaseLostError, KeyboardInterrupt):
                raise
            except Exception as error:
                LOGGER.exception("Document %s failed", path)
                failures.append(f"{path.name}: {type(error).__name__}: {error}")
            else:
                succeeded += 1
                warnings.extend(result.warnings)
            if lease_lost.is_set():
                raise JobLeaseLostError(f"Job lease lost: {job_id}")
            self.store.update_job(
                job_id,
                progress=index / total,
                owner=self.owner,
            )
        if failures and not succeeded:
            self.store.update_job(
                job_id,
                status="failed",
                stage=f"failed-all-{len(failures)}-documents",
                progress=1.0,
                error="\n".join(failures)[:4000],
                owner=self.owner,
            )
            return
        issue_count = len(warnings) + len(failures)
        self.store.update_job(
            job_id,
            status="succeeded",
            stage=(
                "completed"
                if not issue_count
                else f"completed-with-{issue_count}-issues"
            ),
            progress=1.0,
            error="\n".join(failures)[:4000],
            owner=self.owner,
        )

    def _resolve_path(self, relative_path: str) -> Path:
        relative = Path(relative_path.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Job path must be relative to the corpus root")
        root = self.settings.corpus_root.resolve()
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise ValueError("Job path escapes the corpus root") from error
        if not target.exists():
            raise FileNotFoundError(f"Corpus path does not exist: {relative_path}")
        return target


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="secknow-worker")
    parser.add_argument("--once", action="store_true", help="process at most one queued job")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    worker = Worker(get_settings())

    def stop(_signum: int, _frame: FrameType | None) -> None:
        worker.request_stop()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    if args.once:
        worker.run_once()
    else:
        worker.run_forever(args.poll_seconds)


if __name__ == "__main__":
    run()
