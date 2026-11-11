from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from secknow.domain.ids import canonical_json_hash, normalize_text, sha256_file
from secknow.domain.models import SourceRecord
from secknow.sources.identity import canonical_native_id, record_identity
from secknow.sources.manifest import LicenseInfo, SourceEntry


class AdapterContext:
    __slots__ = (
        "license",
        "observed_at",
        "provenance_urls",
        "snapshot_id",
        "source_id",
        "source_revision",
    )

    def __init__(
        self,
        *,
        source_id: str,
        source_revision: str,
        license: LicenseInfo,
        snapshot_id: str | None = None,
        provenance_urls: Sequence[str] = (),
        observed_at: datetime | None = None,
    ) -> None:
        self.source_id = source_id
        self.source_revision = source_revision
        self.license = license
        self.snapshot_id = snapshot_id
        self.provenance_urls = tuple(provenance_urls)
        self.observed_at = observed_at or datetime.now(UTC)

    @classmethod
    def from_source(
        cls,
        source: SourceEntry,
        *,
        source_revision: str,
        snapshot_id: str | None = None,
        observed_at: datetime | None = None,
    ) -> AdapterContext:
        return cls(
            source_id=source.id,
            source_revision=source_revision,
            license=source.license,
            snapshot_id=snapshot_id,
            provenance_urls=tuple(str(location.url) for location in source.locations),
            observed_at=observed_at,
        )


class BaseSourceAdapter(ABC):
    adapter_name: str
    suffixes: tuple[str, ...]

    def load(self, path: str | Path, context: AdapterContext) -> tuple[SourceRecord, ...]:
        return tuple(self.iter_records(Path(path), context))

    @abstractmethod
    def iter_records(self, path: Path, context: AdapterContext) -> Iterator[SourceRecord]:
        raise NotImplementedError

    def files(self, path: Path) -> Iterator[Path]:
        if path.is_file():
            if not self.suffixes or path.suffix.lower() in self.suffixes:
                yield path
            return
        root = path.resolve()
        for candidate in sorted(path.rglob("*")):
            if candidate.is_symlink() or not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root):
                continue
            if not self.suffixes or candidate.suffix.lower() in self.suffixes:
                yield candidate

    def record(
        self,
        *,
        context: AdapterContext,
        path: Path,
        native_id: str,
        title: str,
        body: str,
        status: str = "active",
        language: str = "en",
        aliases: Iterable[str] = (),
        references: Iterable[str] = (),
        modified_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SourceRecord:
        canonical_id = canonical_native_id(native_id)
        clean_title = normalize_text(title) or canonical_id
        clean_body = normalize_text(body)
        unique_aliases = tuple(
            dict.fromkeys(item for item in aliases if item and item != canonical_id)
        )
        unique_references = tuple(dict.fromkeys(item for item in references if item))
        normalized = {
            "native_id": canonical_id,
            "title": clean_title,
            "body": clean_body,
            "status": status,
            "language": language,
            "aliases": unique_aliases,
            "references": unique_references,
        }
        record_metadata: dict[str, Any] = {
            "adapter": self.adapter_name,
            "input_path": path.name,
            "provenance_urls": context.provenance_urls,
            "license": context.license.model_dump(mode="json"),
        }
        if context.snapshot_id:
            record_metadata["snapshot_id"] = context.snapshot_id
        if metadata:
            record_metadata.update(metadata)
        return SourceRecord(
            record_id=record_identity(context.source_id, canonical_id),
            native_id=canonical_id,
            source_id=context.source_id,
            source_revision=context.source_revision,
            title=clean_title,
            body=clean_body,
            status=status,
            language=language,
            aliases=unique_aliases,
            references=unique_references,
            license=context.license.expression,
            raw_blob_sha256=sha256_file(path),
            normalized_sha256=canonical_json_hash(normalized),
            modified_at=modified_at,
            observed_at=context.observed_at,
            metadata=record_metadata,
        )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def text_values(items: object, *, key: str = "value") -> tuple[str, ...]:
    if not isinstance(items, list):
        return ()
    values: list[str] = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get(key), str):
            values.append(item[key])
    return tuple(values)


def parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = f"{candidate[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        try:
            parsed = datetime.strptime(candidate, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
