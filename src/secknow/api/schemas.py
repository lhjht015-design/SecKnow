from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from secknow.config import RuntimeMode, Settings, get_settings
from secknow.stores.metadata import MetadataStore


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchRequest(ApiModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    mode: RuntimeMode | None = None
    use_llm: bool = False

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query cannot be empty")
        return value


class IndexRequest(ApiModel):
    path: str = Field(default=".", min_length=1, max_length=1024)
    recursive: bool = True

    @field_validator("path")
    @classmethod
    def relative_path_only(cls, value: str) -> str:
        value = value.strip().replace("\\", "/")
        candidate = Path(value)
        if not value or candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("path must be relative to the corpus root")
        return value


class LlmConfigRequest(ApiModel):
    enabled: bool
    base_url: str = Field(min_length=1, max_length=2048)
    model: str = Field(min_length=1, max_length=256)
    api_key: str | None = Field(default=None, max_length=4096)


class SnapshotPublishRequest(ApiModel):
    allow_quarantine: bool = False


class SnapshotRollbackRequest(ApiModel):
    snapshot_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class JobAccepted(ApiModel):
    job_id: str
    status: Literal["queued"] = "queued"


class ErrorDetail(ApiModel):
    code: str
    message: str


class ErrorResponse(ApiModel):
    detail: ErrorDetail


@lru_cache(maxsize=8)
def _store_for_path(path: str) -> MetadataStore:
    store = MetadataStore(Path(path))
    store.initialize()
    return store


def request_settings(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    return settings if isinstance(settings, Settings) else get_settings()


def request_store(settings: Annotated[Settings, Depends(request_settings)]) -> MetadataStore:
    return _store_for_path(str(settings.database_path))


SettingsDependency = Annotated[Settings, Depends(request_settings)]
StoreDependency = Annotated[MetadataStore, Depends(request_store)]
