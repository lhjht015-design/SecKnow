"""pwntools 适配层：常驻 WSL worker，把 gadget/ROP/执行封装成可复用接口。

设计约束
    - 只由求解侧显式调用；产品 API 不执行目标程序。
    - worker 在 WSL 内常驻，避免每次尝试都起一个 `wsl -e` 进程。
    - 所有失败都转成 ToolUnavailable / 结构化结果，不静默。
"""

from __future__ import annotations

import contextlib
import json
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any

DEFAULT_WORKER = Path(__file__).resolve().parents[4] / "tools" / "pwn_worker.src"


class ToolUnavailable(RuntimeError):
    pass


def _wsl_path(win_path: str | Path) -> str:
    path = Path(win_path).resolve()
    drive = path.drive.rstrip(":").lower()
    return f"/mnt/{drive}" + str(path)[2:].replace("\\", "/")


class PwntoolsWorker:
    def __init__(
        self,
        worker_path: Path | None = None,
        python: str = "python3",
        startup_timeout: float = 20.0,
        request_timeout: float = 120.0,
    ) -> None:
        self.worker_path = Path(worker_path or DEFAULT_WORKER)
        self.python = python
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self._process: subprocess.Popen[str] | None = None
        self._replies: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._next_id = 0
        self._version: str | None = None
        self._checked = False

    # ------------------------------------------------------------------ lifecycle

    def _start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        try:
            self._process = subprocess.Popen(
                ["wsl", "-e", self.python, _wsl_path(self.worker_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            raise ToolUnavailable(f"worker_start_failed:{error}") from error
        self._replies = queue.Queue()
        thread = threading.Thread(target=self._reader, args=(self._process, self._replies),
                                  daemon=True)
        thread.start()

    @staticmethod
    def _reader(process: subprocess.Popen[str], replies: queue.Queue[str | None]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            replies.put(line.strip())
        replies.put(None)

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    def __del__(self) -> None:  # pragma: no cover - 析构兜底
        with contextlib.suppress(Exception):
            self.close()

    # ------------------------------------------------------------------ protocol

    def _request(self, op: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            self._start()
            assert self._process is not None and self._process.stdin is not None
            self._next_id += 1
            request_id = self._next_id
            message = {"id": request_id, "op": op, **payload}
            try:
                self._process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                self.close()
                raise ToolUnavailable(f"worker_write_failed:{error}") from error
            try:
                line = self._replies.get(timeout=self.request_timeout)
            except queue.Empty as error:
                self.close()
                raise ToolUnavailable("worker_timeout") from error
            if line is None:
                self.close()
                raise ToolUnavailable("worker_exited")
        try:
            response = json.loads(line)
        except ValueError as error:
            raise ToolUnavailable(f"worker_bad_reply:{line[:120]}") from error
        if response.get("id") != request_id:
            raise ToolUnavailable("worker_reply_id_mismatch")
        if not response.get("ok"):
            raise ToolUnavailable(str(response.get("error") or "worker_error"))
        return response.get("data") or {}

    # ------------------------------------------------------------------ public API

    def available(self, timeout: float | None = None) -> bool:
        if self._checked:
            return self._version is not None
        self._checked = True
        previous = self.request_timeout
        if timeout is not None:
            self.request_timeout = timeout
        try:
            data = self._request("ping")
            self._version = str(data.get("version") or "")
        except ToolUnavailable:
            self._version = None
        finally:
            self.request_timeout = previous
        return self._version is not None

    @property
    def version(self) -> str | None:
        return self._version

    def gadgets(self, binary: str | Path) -> dict[str, Any]:
        return self._request("gadgets", path=_wsl_path(binary))

    def facts(self, binary: str | Path, search: list[str] | None = None) -> dict[str, Any]:
        return self._request("facts", path=_wsl_path(binary), search=search or ["/bin/sh"])

    def run_chain(self, binary: str | Path, payloads: list[bytes], post: bytes = b"",
                  marker: str = "", timeout: float = 3.0) -> dict[str, Any]:
        return self._request("run_chain", path=_wsl_path(binary),
                             payloads=[payload.hex() for payload in payloads],
                             post=post.hex(), marker=marker, timeout=timeout)

    def build(self, binary: str | Path, target: int, mode: str,
              arg: int, offset: int) -> dict[str, Any]:
        return self._request("build", path=_wsl_path(binary), target=target,
                             mode=mode, arg=arg, offset=offset)

    def run(self, binary: str | Path, target: int, mode: str, arg: int,
            offsets: list[int], flag: str, timeout: float = 5.0) -> dict[str, Any]:
        return self._request("run", path=_wsl_path(binary), target=target,
                             mode=mode, arg=arg, offsets=offsets, flag=flag,
                             timeout=timeout)
