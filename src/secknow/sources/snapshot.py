from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, Self, TypedDict

from pydantic import Field, field_validator

from secknow.domain.ids import canonical_json_hash, sha256_bytes, sha256_file
from secknow.sources.manifest import LicenseInfo, RiskState, StrictModel
from secknow.sources.validation import confined_path, safe_relative_path

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class SnapshotState(StrEnum):
    STAGING = "staging"
    QUARANTINE = "quarantine"
    NORMALIZED = "normalized"
    PUBLISHED = "published"


class SnapshotFile(StrictModel):
    relative_path: str
    blob_sha256: str = Field(pattern=_SHA256_PATTERN)
    size_bytes: int = Field(ge=0)
    media_type: str = Field(pattern=r"^[\w.+-]+/[\w.+-]+$")

    @field_validator("relative_path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return safe_relative_path(value).as_posix()


class SnapshotMetadata(StrictModel):
    snapshot_id: str = Field(pattern=_SHA256_PATTERN)
    source_id: str
    source_revision: str = Field(min_length=1)
    created_at: datetime
    risk_state: RiskState
    license: LicenseInfo
    files: tuple[SnapshotFile, ...]
    provenance_urls: tuple[str, ...] = ()

    @field_validator("files")
    @classmethod
    def unique_paths(cls, value: tuple[SnapshotFile, ...]) -> tuple[SnapshotFile, ...]:
        paths = [item.relative_path.casefold() for item in value]
        if len(paths) != len(set(paths)):
            raise ValueError("snapshot file paths must be unique")
        return value

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        source_revision: str,
        risk_state: RiskState,
        license: LicenseInfo,
        files: tuple[SnapshotFile, ...],
        provenance_urls: tuple[str, ...] = (),
        created_at: datetime | None = None,
    ) -> Self:
        timestamp = created_at or datetime.now(UTC)
        identity = {
            "source_id": source_id,
            "source_revision": source_revision,
            "risk_state": risk_state,
            "license": license.model_dump(mode="json"),
            "files": [item.model_dump(mode="json") for item in files],
            "provenance_urls": provenance_urls,
        }
        return cls(
            snapshot_id=canonical_json_hash(identity),
            source_id=source_id,
            source_revision=source_revision,
            created_at=timestamp,
            risk_state=risk_state,
            license=license,
            files=files,
            provenance_urls=provenance_urls,
        )


class BlobStore:
    """Content-addressed immutable blob storage with governed lifecycle directories."""

    directory_names = ("blobs", "staging", "quarantine", "normalized", "published")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        for name in self.directory_names:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def path_for(self, digest: str) -> Path:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("invalid SHA-256 digest")
        return self.root / "blobs" / digest[:2] / digest[2:]

    def put_bytes(self, data: bytes) -> str:
        digest = sha256_bytes(data)
        destination = self.path_for(digest)
        if destination.exists():
            return digest
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=".blob-", dir=self.root / "staging")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if not destination.exists():
                os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return digest

    def put_file(self, path: str | Path) -> str:
        source = Path(path)
        digest = sha256_file(source)
        destination = self.path_for(digest)
        if destination.exists():
            return digest
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=".blob-", dir=self.root / "staging")
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            if sha256_file(temporary) != digest:
                raise OSError("source changed while being copied into the blob store")
            if not destination.exists():
                os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return digest

    def put(self, value: bytes | str | Path | BinaryIO) -> str:
        if isinstance(value, bytes):
            return self.put_bytes(value)
        if isinstance(value, (str, Path)):
            return self.put_file(value)
        return self.put_bytes(value.read())

    def verify(self, digest: str) -> bool:
        path = self.path_for(digest)
        return path.is_file() and sha256_file(path) == digest


class CatalogEntry(TypedDict):
    current: str | None
    history: list[str]


class CatalogData(TypedDict):
    version: int
    sources: dict[str, CatalogEntry]


class SnapshotCatalog:
    """Atomically stages, publishes, and rolls back immutable snapshots."""

    def __init__(self, root: str | Path, blob_store: BlobStore | None = None) -> None:
        self.root = Path(root)
        self.blobs = blob_store or BlobStore(self.root)
        self.catalog_path = self.root / "catalog.json"
        self._lock = threading.RLock()

    def _snapshot_path(self, state: SnapshotState, metadata: SnapshotMetadata) -> Path:
        return self.root / state / metadata.source_id / metadata.snapshot_id

    @staticmethod
    def _metadata_bytes(metadata: SnapshotMetadata) -> bytes:
        data = metadata.model_dump(mode="json")
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def stage(self, metadata: SnapshotMetadata) -> Path:
        target_state = (
            SnapshotState.QUARANTINE
            if metadata.risk_state == RiskState.QUARANTINE
            else SnapshotState.NORMALIZED
        )
        destination = self._snapshot_path(target_state, metadata)
        if destination.exists():
            existing = self.read_metadata(destination)
            if existing != metadata:
                raise ValueError("snapshot id collision with different metadata")
            return destination
        for item in metadata.files:
            blob = self.blobs.path_for(item.blob_sha256)
            if not self.blobs.verify(item.blob_sha256) or blob.stat().st_size != item.size_bytes:
                raise ValueError(f"missing or invalid blob: {item.blob_sha256}")
        staging_parent = self.root / SnapshotState.STAGING / metadata.source_id
        staging_parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{metadata.snapshot_id}-", dir=staging_parent))
        try:
            payload = temporary / "payload"
            payload.mkdir()
            for item in metadata.files:
                output = confined_path(payload, item.relative_path)
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.blobs.path_for(item.blob_sha256), output)
            (temporary / "snapshot.json").write_bytes(self._metadata_bytes(metadata))
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, destination)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        return destination

    def publish(self, snapshot_id: str, *, allow_quarantine: bool = False) -> SnapshotMetadata:
        with self._lock:
            source, state = self._find_snapshot(snapshot_id)
            metadata = self.read_metadata(source)
            if state == SnapshotState.QUARANTINE and not allow_quarantine:
                raise PermissionError("quarantined snapshots require explicit publication approval")
            destination = self._snapshot_path(SnapshotState.PUBLISHED, metadata)
            if source != destination:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if self.read_metadata(destination) != metadata:
                        raise ValueError("published snapshot id collision")
                    shutil.rmtree(source)
                else:
                    os.replace(source, destination)
            catalog = self._read_catalog()
            entry = catalog["sources"].setdefault(
                metadata.source_id, {"current": None, "history": []}
            )
            if snapshot_id not in entry["history"]:
                entry["history"].append(snapshot_id)
            entry["current"] = snapshot_id
            self._write_catalog(catalog)
            return metadata

    def current(self, source_id: str) -> SnapshotMetadata | None:
        catalog = self._read_catalog()
        entry = catalog["sources"].get(source_id)
        if not entry or not entry["current"]:
            return None
        path = self.root / SnapshotState.PUBLISHED / source_id / entry["current"]
        return self.read_metadata(path)

    def rollback(self, source_id: str, snapshot_id: str | None = None) -> SnapshotMetadata:
        with self._lock:
            catalog = self._read_catalog()
            entry = catalog["sources"].get(source_id)
            if not entry:
                raise KeyError(source_id)
            history: list[str] = entry["history"]
            if snapshot_id is None:
                current = entry["current"]
                candidates = [item for item in history if item != current]
                if not candidates:
                    raise ValueError("no previous snapshot is available")
                snapshot_id = candidates[-1]
            if snapshot_id not in history:
                raise ValueError("rollback target is not in publication history")
            path = self.root / SnapshotState.PUBLISHED / source_id / snapshot_id
            if not path.is_dir():
                raise FileNotFoundError(path)
            entry["current"] = snapshot_id
            self._write_catalog(catalog)
            return self.read_metadata(path)

    def state(self, snapshot_id: str) -> SnapshotState:
        return self._find_snapshot(snapshot_id)[1]

    def read_metadata(self, snapshot_directory: str | Path) -> SnapshotMetadata:
        path = Path(snapshot_directory) / "snapshot.json"
        return SnapshotMetadata.model_validate_json(path.read_bytes(), strict=True)

    def _find_snapshot(self, snapshot_id: str) -> tuple[Path, SnapshotState]:
        for state in (
            SnapshotState.NORMALIZED,
            SnapshotState.QUARANTINE,
            SnapshotState.PUBLISHED,
        ):
            state_root = self.root / state
            for path in state_root.glob(f"*/{snapshot_id}"):
                if path.is_dir():
                    return path, state
        raise KeyError(snapshot_id)

    def _read_catalog(self) -> CatalogData:
        if not self.catalog_path.exists():
            return {"version": 1, "sources": {}}
        data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("invalid snapshot catalog")
        raw_sources = data.get("sources")
        if not isinstance(raw_sources, dict):
            raise ValueError("invalid snapshot catalog sources")
        sources: dict[str, CatalogEntry] = {}
        for source_id, raw_entry in raw_sources.items():
            if not isinstance(source_id, str) or not isinstance(raw_entry, dict):
                raise ValueError("invalid snapshot catalog entry")
            current = raw_entry.get("current")
            history = raw_entry.get("history")
            if current is not None and not isinstance(current, str):
                raise ValueError("invalid current snapshot id")
            if not isinstance(history, list) or not all(
                isinstance(item, str) for item in history
            ):
                raise ValueError("invalid snapshot history")
            sources[source_id] = {"current": current, "history": list(history)}
        return {"version": 1, "sources": sources}

    def _write_catalog(self, catalog: CatalogData) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=".catalog-", dir=self.root)
        temporary = Path(temporary_name)
        try:
            encoded = json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode("utf-8")
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.catalog_path)
        finally:
            temporary.unlink(missing_ok=True)
