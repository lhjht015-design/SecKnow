from __future__ import annotations

import asyncio
import hashlib
import json
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def model_directory_ready(model_path: Path) -> bool:
    """Return whether a local Transformers model directory contains config and weights."""
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        return False
    weight_patterns = ("*.safetensors", "*.bin", "*.pt")
    weights = [path for pattern in weight_patterns for path in model_path.glob(pattern)]
    revision_path = model_path / "REVISION"
    checksum_path = model_path / "SHA256SUMS"
    if not weights or not revision_path.is_file() or not checksum_path.is_file():
        return False
    try:
        revision = revision_path.read_text(encoding="ascii").strip()
        checksums = checksum_path.read_text(encoding="ascii")
    except OSError:
        return False
    return bool(re.fullmatch(r"[0-9a-f]{40}", revision)) and all(
        weight.relative_to(model_path).as_posix() in checksums for weight in weights
    )


class BgeM3Encoder:
    def __init__(
        self,
        model_path: Path,
        device: str = "auto",
        batch_size: int = 4,
        max_length: int = 512,
        cache_dir: Path | None = None,
        model_revision: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.model_revision = model_revision
        self._model_fingerprint: str | None = None
        self._model: Any = None

    @property
    def available(self) -> bool:
        return model_directory_ready(self.model_path)

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.available:
            raise FileNotFoundError(f"BGE-M3 model is not installed: {self.model_path}")
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as error:
            raise RuntimeError("Install SecKnow with the 'semantic' extra") from error
        use_fp16 = self.device != "cpu"
        self._model = BGEM3FlagModel(
            str(self.model_path),
            use_fp16=use_fp16,
            devices=None if self.device == "auto" else self.device,
        )

    def encode(self, texts: Sequence[str]) -> tuple[np.ndarray, list[dict[int, float]]]:
        if self.cache_dir is None:
            return self._encode_uncached(texts)
        dense_rows: list[np.ndarray | None] = [None] * len(texts)
        sparse_rows: list[dict[int, float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        for index, text in enumerate(texts):
            cached = self._load_cached(text)
            if cached is None:
                missing_indices.append(index)
                missing_texts.append(text)
            else:
                dense_rows[index], sparse_rows[index] = cached
        if missing_texts:
            missing_dense, missing_sparse = self._encode_uncached(missing_texts)
            for position, index in enumerate(missing_indices):
                dense = missing_dense[position]
                sparse = missing_sparse[position]
                dense_rows[index] = dense
                sparse_rows[index] = sparse
                self._save_cached(missing_texts[position], dense, sparse)
        if not texts:
            return np.empty((0, 1024), dtype=np.float32), []
        if any(row is None for row in dense_rows) or any(row is None for row in sparse_rows):
            raise RuntimeError("BGE-M3 cache assembly failed")
        return (
            np.stack([row for row in dense_rows if row is not None]),
            [row for row in sparse_rows if row is not None],
        )

    def _encode_uncached(
        self,
        texts: Sequence[str],
    ) -> tuple[np.ndarray, list[dict[int, float]]]:
        self.load()
        output = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = np.asarray(output["dense_vecs"], dtype=np.float32)
        sparse = [
            {int(token): float(weight) for token, weight in weights.items()}
            for weights in output["lexical_weights"]
        ]
        return dense, sparse

    def _cache_path(self, text: str) -> Path:
        namespace = f"{self._fingerprint()}\0{self.max_length}\0{text}"
        digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
        if self.cache_dir is None:
            raise RuntimeError("BGE-M3 cache is disabled")
        return self.cache_dir / digest[:2] / f"{digest}.npz"

    def _fingerprint(self) -> str:
        if self._model_fingerprint is not None:
            return self._model_fingerprint
        if self.model_revision:
            fingerprint = f"revision:{self.model_revision}"
        elif not self.model_path.is_dir():
            fingerprint = f"path:{self.model_path.resolve()}"
        else:
            entries: list[tuple[str, int, int]] = []
            for path in sorted(self.model_path.rglob("*")):
                if not path.is_file():
                    continue
                stat = path.stat()
                entries.append(
                    (
                        path.relative_to(self.model_path).as_posix(),
                        stat.st_size,
                        stat.st_mtime_ns,
                    )
                )
            payload = json.dumps(entries, ensure_ascii=True, separators=(",", ":"))
            fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self._model_fingerprint = fingerprint
        return fingerprint

    def _load_cached(self, text: str) -> tuple[np.ndarray, dict[int, float]] | None:
        path = self._cache_path(text)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as payload:
                dense = np.asarray(payload["dense"], dtype=np.float32)
                token_ids = np.asarray(payload["token_ids"], dtype=np.int64)
                weights = np.asarray(payload["weights"], dtype=np.float32)
            if dense.ndim != 1 or token_ids.ndim != 1 or weights.ndim != 1:
                return None
            if len(token_ids) != len(weights):
                return None
            return dense, {
                int(token): float(weight)
                for token, weight in zip(token_ids, weights, strict=True)
            }
        except (KeyError, OSError, ValueError):
            return None

    def _save_cached(
        self,
        text: str,
        dense: np.ndarray,
        sparse: dict[int, float],
    ) -> None:
        path = self._cache_path(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                suffix=".npz",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                token_ids = np.fromiter(sparse, dtype=np.int64)
                weights = np.fromiter(sparse.values(), dtype=np.float32)
                np.savez_compressed(
                    stream,
                    dense=np.asarray(dense, dtype=np.float32),
                    token_ids=token_ids,
                    weights=weights,
                )
            temporary_path.replace(path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    async def encode_async(self, texts: Sequence[str]) -> tuple[np.ndarray, list[dict[int, float]]]:
        return await asyncio.to_thread(self.encode, texts)


class BgeReranker:
    def __init__(self, model_path: Path, device: str = "auto") -> None:
        self.model_path = model_path
        self.device = device
        self._model: Any = None

    @property
    def available(self) -> bool:
        return model_directory_ready(self.model_path)

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        if not self.available:
            raise FileNotFoundError(f"Reranker is not installed: {self.model_path}")
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as error:
                raise RuntimeError("Install SecKnow with the 'semantic' extra") from error
            self._model = FlagReranker(
                str(self.model_path),
                use_fp16=self.device != "cpu",
                devices=None if self.device == "auto" else self.device,
            )
        scores = self._model.compute_score([[query, passage] for passage in passages])
        if isinstance(scores, float):
            return [scores]
        return [float(score) for score in scores]
