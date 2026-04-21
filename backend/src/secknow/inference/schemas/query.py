from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SemanticQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    zone_id: str
    top_k: int = Field(default=10, gt=0)
    filters: dict[str, str | int | float | bool] = Field(default_factory=dict)


class CodeScanQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code_snippet: str = Field(min_length=1)
    lang: str = Field(min_length=1)
    zone_id: str = "cyber"
    top_k: int = Field(default=10, gt=0)
