from __future__ import annotations

import json
import re
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, TypeAdapter, field_validator

_SOURCE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_LICENSE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]*$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RiskState(StrEnum):
    DEFAULT = "default"
    OPT_IN = "opt-in"
    QUARANTINE = "quarantine"


class LocationKind(StrEnum):
    FILE = "file"
    ARCHIVE = "archive"
    GIT = "git"
    INDEX = "index"


class ArchiveFormat(StrEnum):
    ZIP = "zip"
    TAR = "tar"
    GZIP = "gzip"


class LicenseInfo(StrictModel):
    expression: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=256)
    url: HttpUrl
    attribution_required: bool
    redistribution_allowed: bool

    @field_validator("expression")
    @classmethod
    def valid_expression(cls, value: str) -> str:
        if not _LICENSE_ID.fullmatch(value):
            raise ValueError("license expression must be an SPDX id or LicenseRef identifier")
        return value


class SourceLocation(StrictModel):
    url: HttpUrl
    kind: LocationKind
    media_type: str = Field(pattern=r"^[\w.+-]+/[\w.+-]+$")
    archive: ArchiveFormat | None = None

    @field_validator("url")
    @classmethod
    def https_only(cls, value: HttpUrl) -> HttpUrl:
        if value.scheme != "https":
            raise ValueError("source locations must use HTTPS")
        return value

    @field_validator("archive")
    @classmethod
    def archive_requires_archive_kind(
        cls, value: ArchiveFormat | None, info: object
    ) -> ArchiveFormat | None:
        data = getattr(info, "data", {})
        if value is not None and data.get("kind") != LocationKind.ARCHIVE:
            raise ValueError("archive format is only valid for archive locations")
        if value is None and data.get("kind") == LocationKind.ARCHIVE:
            raise ValueError("archive locations require an archive format")
        return value


class SourceEntry(StrictModel):
    id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=256)
    description: str = Field(min_length=1, max_length=1024)
    homepage: HttpUrl
    adapter: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    risk_state: RiskState
    license: LicenseInfo
    locations: tuple[SourceLocation, ...] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not _SOURCE_ID.fullmatch(value):
            raise ValueError("source id must be lower-case kebab-case")
        return value


class Manifest(StrictModel):
    version: int = Field(ge=1)
    sources: tuple[SourceEntry, ...] = Field(min_length=1)

    @field_validator("sources")
    @classmethod
    def unique_sources(cls, value: tuple[SourceEntry, ...]) -> tuple[SourceEntry, ...]:
        ids = [entry.id for entry in value]
        if len(ids) != len(set(ids)):
            raise ValueError("source ids must be unique")
        return value

    def by_id(self, source_id: str) -> SourceEntry:
        for source in self.sources:
            if source.id == source_id:
                return source
        raise KeyError(source_id)

    def enabled(self, *, include_opt_in: bool = False) -> tuple[SourceEntry, ...]:
        states = {RiskState.DEFAULT}
        if include_opt_in:
            states.add(RiskState.OPT_IN)
        return tuple(source for source in self.sources if source.risk_state in states)


def _manifest_input(raw: dict[str, object]) -> dict[str, object]:
    """Preserve strict validation while converting YAML collection syntax."""
    converted = dict(raw)
    sources = converted.get("sources")
    if not isinstance(sources, list):
        return converted
    converted_sources: list[object] = []
    for item in sources:
        if not isinstance(item, dict):
            converted_sources.append(item)
            continue
        source = dict(item)
        locations = source.get("locations")
        if isinstance(locations, list):
            source["locations"] = tuple(locations)
        risk_state = source.get("risk_state")
        if isinstance(risk_state, str):
            source["risk_state"] = RiskState(risk_state)
        converted_sources.append(source)
    converted["sources"] = tuple(converted_sources)
    return converted


def load_manifest(path: str | Path) -> Manifest:
    manifest_path = Path(path)
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid source manifest YAML: {manifest_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("source manifest root must be a mapping")
    return TypeAdapter(Manifest).validate_json(
        json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
        strict=True,
    )
