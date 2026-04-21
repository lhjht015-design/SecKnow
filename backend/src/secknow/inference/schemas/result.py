from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    zone_id: str
    score: float
    text: str
    filename: str | None = None
    source_path: str | None = None
    record_type: str | None = None


class CodeRiskItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code_unit: str
    summary: str
    severity: str = "info"
    references: list[SearchResult] = Field(default_factory=list)


class CodeRiskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lang: str
    zone_id: str
    items: list[CodeRiskItem] = Field(default_factory=list)
