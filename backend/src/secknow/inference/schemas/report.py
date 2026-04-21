from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .result import CodeRiskResult, SearchResult


class SearchReport(BaseModel):
    """面向展示层的语义检索报告。"""

    model_config = ConfigDict(extra="forbid")

    query: str
    total: int
    results: list[SearchResult] = Field(default_factory=list)


class CodeRiskReport(BaseModel):
    """面向展示层的代码风险扫描报告。"""

    model_config = ConfigDict(extra="forbid")

    total: int
    result: CodeRiskResult
