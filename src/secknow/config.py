from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeMode(StrEnum):
    STRICT_LEXICAL = "strict-lexical"
    COMPETITION = "competition"
    ENHANCED = "enhanced"
    LLM = "llm"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SECKNOW_",
        extra="ignore",
        case_sensitive=False,
    )

    mode: RuntimeMode = RuntimeMode.COMPETITION
    data_root: Path = Path("data")
    corpus_root: Path = Path("book")
    database_path: Path = Path("data/secknow.db")
    snapshot_root: Path = Path("data/snapshots")
    model_root: Path = Path("models")
    frontend_dist: Path = Path("frontend/dist")
    source_manifest: Path = Path("sources/sources.yaml")

    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1024, le=65535)
    workers: int = Field(default=1, ge=1, le=4)
    max_upload_mb: int = Field(default=256, ge=1, le=2048)
    max_pages: int = Field(default=5000, ge=1)
    chunk_size: int = Field(default=480, ge=128, le=1024)
    chunk_overlap: int = Field(default=64, ge=0, le=256)
    retrieval_top_k: int = Field(default=40, ge=5, le=200)
    reranker_top_k: int = Field(default=24, ge=1, le=100)
    result_top_k: int = Field(default=12, ge=1, le=50)
    graph_max_nodes: int = Field(default=160, ge=20, le=1000)
    graph_max_hops: int = Field(default=2, ge=1, le=4)
    semantic_edge_top_k: int = Field(default=3, ge=1, le=20)
    semantic_edge_threshold: float = Field(default=0.72, ge=0, le=1)
    semantic_edge_block_size: int = Field(default=256, ge=16, le=2048)

    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: SecretStr | None = None
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("change-me-before-enhanced-mode")

    bge_model_path: Path = Path("models/bge-m3")
    bge_model_revision: str | None = None
    reranker_model_path: Path = Path("models/bge-reranker-v2-m3")
    tesseract_executable: Path | None = None
    tesseract_tessdata_dir: Path | None = None
    # OCR 后端：auto 优先 RapidOCR（纯 pip、无需外部可执行文件），退回 Tesseract
    ocr_backend: str = "auto"
    ocr_languages: str = "chi_sim+eng"
    ocr_dpi: int = Field(default=240, ge=144, le=400)
    device: str = "auto"
    embedding_batch_size: int = Field(default=4, ge=1, le=128)
    reranker_enabled: bool = True

    llm_enabled: bool = False
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "phi4"
    llm_api_key: SecretStr = SecretStr("local-only")
    llm_timeout_seconds: float = Field(default=120.0, ge=1, le=600)

    source_download_budget_gb: float = Field(default=20.0, ge=0.1, le=500)
    offline: bool = True

    @field_validator(
        "data_root",
        "corpus_root",
        "database_path",
        "snapshot_root",
        "model_root",
        "frontend_dist",
        "source_manifest",
        "bge_model_path",
        "reranker_model_path",
        "tesseract_executable",
        "tesseract_tessdata_dir",
        mode="after",
    )
    @classmethod
    def normalize_path(cls, value: Path | None) -> Path | None:
        return value.expanduser().resolve() if value is not None else None

    @property
    def generative_enabled(self) -> bool:
        return self.mode is RuntimeMode.LLM and self.llm_enabled

    @property
    def semantic_enabled(self) -> bool:
        return self.mode is not RuntimeMode.STRICT_LEXICAL

    @property
    def infrastructure_enabled(self) -> bool:
        return self.mode in {RuntimeMode.COMPETITION, RuntimeMode.ENHANCED, RuntimeMode.LLM}

    def ensure_directories(self) -> None:
        for directory in (self.data_root, self.snapshot_root, self.database_path.parent):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
