from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .result import CodeRiskResult, SearchResult


class SearchReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    total: int
    results: list[SearchResult] = Field(default_factory=list)


class CodeRiskReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int
    result: CodeRiskResult
