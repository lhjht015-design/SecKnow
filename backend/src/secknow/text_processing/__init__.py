from __future__ import annotations

"""4.1 文本处理模块对外入口。"""

from .facade import (
    DocumentTextPipeline,
    chunk_text,
    dedup,
    encode_chunks,
    load_document,
    run_pipeline,
)

__all__ = [
    "DocumentTextPipeline",
    "load_document",
    "chunk_text",
    "dedup",
    "encode_chunks",
    "run_pipeline",
]
