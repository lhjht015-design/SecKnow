from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DocumentFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    TEXT = "txt"
    PYTHON = "py"
    MARKDOWN = "md"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    STIX = "stix"


class ElementKind(StrEnum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE = "code"
    IMAGE = "image"
    PAGE_BREAK = "page_break"
    UNKNOWN = "unknown"


class BoundingBox(FrozenModel):
    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float | None = None
    page_height: float | None = None


class Citation(FrozenModel):
    source_id: str
    source_version: str
    title: str
    relative_path: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    section_path: tuple[str, ...] = ()
    url: HttpUrl | None = None
    bounding_boxes: tuple[BoundingBox, ...] = ()
    extraction_confidence: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def ordered_ranges(self) -> Citation:
        if self.page_start and self.page_end and self.page_end < self.page_start:
            raise ValueError("page_end cannot precede page_start")
        if self.line_start and self.line_end and self.line_end < self.line_start:
            raise ValueError("line_end cannot precede line_start")
        return self


class SourceDocument(FrozenModel):
    source_id: str
    version_id: str
    title: str
    relative_path: str
    absolute_path: str
    format: DocumentFormat
    media_type: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    language: str | None = None
    modified_at: datetime | None = None
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedElement(FrozenModel):
    element_id: str
    source_id: str
    source_version: str
    kind: ElementKind
    text: str
    order: int = Field(ge=0)
    page: int | None = Field(default=None, ge=1)
    section_path: tuple[str, ...] = ()
    bounding_box: BoundingBox | None = None
    language: str | None = None
    parser: str
    parser_version: str | None = None
    extraction_confidence: float = Field(default=1.0, ge=0, le=1)
    diagnostics: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class CodeSymbol(FrozenModel):
    symbol_id: str
    element_id: str
    name: str
    qualified_name: str
    kind: str
    language: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    parent_symbol_id: str | None = None
    signature: str | None = None
    docstring: str | None = None
    imports: tuple[str, ...] = ()
    calls: tuple[str, ...] = ()


class Chunk(FrozenModel):
    chunk_id: str
    source_id: str
    source_version: str
    text: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    element_ids: tuple[str, ...]
    order: int = Field(ge=0)
    token_count: int = Field(ge=1)
    section_path: tuple[str, ...] = ()
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    citation: Citation
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceAssertion(FrozenModel):
    assertion_id: str
    source_record_id: str
    predicate: str
    value: Any
    authority: str
    source_revision: str
    confidence: float = Field(default=1.0, ge=0, le=1)
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SourceRecord(FrozenModel):
    record_id: str
    native_id: str
    source_id: str
    source_revision: str
    title: str
    body: str
    status: str = "active"
    language: str = "en"
    aliases: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    license: str
    raw_blob_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assertions: tuple[SourceAssertion, ...] = ()
    modified_at: datetime | None = None
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Entity(FrozenModel):
    entity_id: str
    canonical_name: str
    kind: str
    aliases: tuple[str, ...] = ()
    confidence: float = Field(default=1.0, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Relation(FrozenModel):
    relation_id: str
    source_id: str
    target_id: str
    kind: str
    weight: float = Field(default=1.0, ge=0)
    confidence: float = Field(default=1.0, ge=0, le=1)
    provenance: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchHit(FrozenModel):
    chunk: Chunk
    score: float
    rank: int = Field(ge=1)
    channels: dict[str, float] = Field(default_factory=dict)
    graph_paths: tuple[tuple[str, ...], ...] = ()
    explanation: str | None = None


class SearchResponse(FrozenModel):
    query: str
    mode: str
    hits: tuple[SearchHit, ...]
    elapsed_ms: float = Field(ge=0)
    ranking_score_calibrated: bool = False
    warnings: tuple[str, ...] = ()
