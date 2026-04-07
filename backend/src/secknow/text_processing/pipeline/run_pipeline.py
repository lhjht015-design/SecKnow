from __future__ import annotations

"""最小闭环流水线：文件 -> 文本 -> 分块 -> 去重 -> 编码 -> records。

**相对早期版本的结构改动**（便于 diff 理解）：
1. **清洗**：`process_file` 不再直接 `basic_clean`，改为 `clean_document_text(raw, file_type)`，
   与扩展名对应的业务类型挂钩（见 `cleaners/document_clean.py`）。
2. **复用**：抽出 `_process_clean_document`，`process_file` 与 `process_text` 共享
   「已解析元数据 + 已清洗全文」之后的逻辑，避免分岔维护。
3. **process_text**：不再向磁盘写临时文件再 `process_file`；改为
   `build_loaded_document_for_text` 构造元数据，满足 API/单测「纯内存」场景。

`dedup_strategy` / `chunk_strategy` 的可选值扩展在各自 router 与 `config` 中说明。
"""

from pathlib import Path

from secknow.vector_store.models import ChunkRecord, RecordType, ZoneId

from ..chunkers.router import chunk_text
from ..cleaners.document_clean import clean_document_text
from ..config import (
    DEFAULT_CHUNK_STRATEGY,
    DEFAULT_DEDUP_STRATEGY,
    get_chunk_max_tokens,
    get_chunk_overlap,
)
from ..dedupers.router import dedup_chunks
from ..encoders.factory import build_embedder
from ..exceptions import ChunkingError
from ..loaders.router import load_document_text
from ..metadata.chunk import build_chunk_metadata
from ..metadata.document import build_loaded_document, build_loaded_document_for_text
from ..schemas.chunk import EncodedChunk, TextChunk
from ..schemas.document import LoadedDocument, RawDocument
from ..schemas.pipeline_result import PipelineResult


class DocumentTextPipeline:
    """4.1 对外主入口类。"""

    def __init__(
        self,
        *,
        chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
        dedup_strategy: str = DEFAULT_DEDUP_STRATEGY,
        max_tokens: int | None = None,
        overlap: int | None = None,
        embedding_mode: str | None = None,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
        embedder=None,
    ):
        self.chunk_strategy = chunk_strategy
        self.dedup_strategy = dedup_strategy
        self.max_tokens = max_tokens or get_chunk_max_tokens()
        self.overlap = get_chunk_overlap() if overlap is None else overlap
        self.embedder = embedder or build_embedder(
            mode=embedding_mode,
            model_name=embedding_model,
            dim=embedding_dim,
        )

    def _process_clean_document(
        self,
        *,
        loaded_document: LoadedDocument,
        raw_text: str,
        clean_text: str,
        zone_id: ZoneId,
        record_type: RecordType,
        language: str | None,
        return_result: bool,
        raw_source_path: str,
    ) -> list[ChunkRecord] | PipelineResult:
        if not clean_text:
            raise ChunkingError(
                f"清洗后文本为空，无法分块: {raw_source_path}"
            )

        # 以下阶段与文件是否来自磁盘无关，只依赖 clean_text + loaded_document。
        chunk_texts = chunk_text(
            clean_text,
            strategy=self.chunk_strategy,
            max_tokens=self.max_tokens,
            overlap=self.overlap,
        )
        original_chunks = [
            TextChunk(text=chunk, chunk_index=i, chunk_count=len(chunk_texts))
            for i, chunk in enumerate(chunk_texts)
        ]

        deduped_texts = dedup_chunks(chunk_texts, strategy=self.dedup_strategy)
        deduped_chunks = [
            TextChunk(text=chunk, chunk_index=i, chunk_count=len(deduped_texts))
            for i, chunk in enumerate(deduped_texts)
        ]

        vectors = self.embedder.encode(deduped_texts)
        encoded_chunks = [
            EncodedChunk(text_chunk=text_chunk, vector=vector)
            for text_chunk, vector in zip(deduped_chunks, vectors, strict=False)
        ]

        records: list[ChunkRecord] = []
        for item in encoded_chunks:
            metadata = build_chunk_metadata(
                document=loaded_document,
                zone_id=zone_id,
                chunk_index=item.text_chunk.chunk_index,
                chunk_count=item.text_chunk.chunk_count,
                chunk_text=item.text_chunk.text,
                record_type=record_type,
                language=language,
            )
            records.append(
                ChunkRecord(
                    text=item.text_chunk.text,
                    vector=item.vector,
                    metadata=metadata,
                )
            )

        if not return_result:
            return records

        return PipelineResult(
            raw_document=RawDocument(source_path=raw_source_path, text=raw_text),
            loaded_document=loaded_document,
            chunks=original_chunks,
            deduped_chunks=deduped_chunks,
            encoded_chunks=encoded_chunks,
            records=records,
        )

    def process_file(
        self,
        file_path: str | Path,
        *,
        zone_id: ZoneId,
        record_type: RecordType = "knowledge",
        language: str | None = None,
        return_result: bool = False,
    ) -> list[ChunkRecord] | PipelineResult:
        """处理文件并输出可直接给 4.3 upsert 的 records。"""
        path = Path(file_path)
        # 先取元数据（含 file_type），再按类型清洗，保证 markdown/code 等走对分支。
        loaded_document = build_loaded_document(path)
        raw_text = load_document_text(path)
        clean_text = clean_document_text(raw_text, loaded_document.file_type)

        return self._process_clean_document(
            loaded_document=loaded_document,
            raw_text=raw_text,
            clean_text=clean_text,
            zone_id=zone_id,
            record_type=record_type,
            language=language,
            return_result=return_result,
            raw_source_path=str(path),
        )

    def process_text(
        self,
        text: str,
        *,
        source_path: str,
        zone_id: ZoneId,
        record_type: RecordType = "knowledge",
        language: str | None = None,
        return_result: bool = False,
    ) -> list[ChunkRecord] | PipelineResult:
        """处理内存中的文本，不写入磁盘。

        `source_path` 仅用于推断扩展名 / 展示用 filename（可指向尚不存在的逻辑路径），
        与旧实现「写临时文件再 process_file」相比，避免副作用与权限问题。
        """
        loaded_document = build_loaded_document_for_text(source_path, text)
        clean_text = clean_document_text(text, loaded_document.file_type)

        return self._process_clean_document(
            loaded_document=loaded_document,
            raw_text=text,
            clean_text=clean_text,
            zone_id=zone_id,
            record_type=record_type,
            language=language,
            return_result=return_result,
            raw_source_path=source_path,
        )


def run_pipeline(
    file_path: str | Path,
    *,
    zone_id: ZoneId,
    record_type: RecordType = "knowledge",
    language: str | None = None,
    pipeline: DocumentTextPipeline | None = None,
) -> list[ChunkRecord]:
    """函数式入口：直接返回 records。"""
    current = pipeline or DocumentTextPipeline()
    return current.process_file(
        file_path,
        zone_id=zone_id,
        record_type=record_type,
        language=language,
        return_result=False,
    )
