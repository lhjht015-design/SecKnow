from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from secknow.config import Settings
from secknow.domain.models import Chunk, CodeSymbol, DocumentFormat, ParsedElement, SourceDocument
from secknow.graph.builder import KnowledgeGraph
from secknow.graph.entities import EntityExtractor
from secknow.indexing.chunker import EmbeddingFunction, SemanticChunker
from secknow.indexing.tfidf import TfidfIndex
from secknow.ingestion.code_parser import CodeParser
from secknow.ingestion.discovery import DocumentDiscovery
from secknow.ingestion.document_parser import (
    PdfExtractor,
    parse_with_unstructured,
    select_ocr_engine,
)
from secknow.ingestion.text_parser import parse_docx, parse_markdown, parse_plain
from secknow.models.bge import BgeM3Encoder
from secknow.stores.metadata import MetadataStore
from secknow.stores.milvus import MilvusStore
from secknow.stores.neo4j import Neo4jStore


@dataclass(frozen=True)
class IngestionResult:
    document: SourceDocument
    elements: tuple[ParsedElement, ...]
    symbols: tuple[CodeSymbol, ...]
    chunks: tuple[Chunk, ...]
    tfidf_version: str
    graph_version: str
    warnings: tuple[str, ...] = ()


def _quarantine_warnings(elements: Sequence[ParsedElement]) -> list[str]:
    return [
        f"page:{element.page}:"
        + "|".join(dict.fromkeys(element.diagnostics or ("quarantined",)))
        for element in elements
        if element.metadata.get("disposition") == "quarantine"
    ]


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        store: MetadataStore | None = None,
        embed: EmbeddingFunction | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or MetadataStore(settings.database_path)
        self.store.initialize()
        self.discovery = DocumentDiscovery(settings.corpus_root)
        self.chunker = SemanticChunker(settings.chunk_size, settings.chunk_overlap)
        self.code_parser = CodeParser()
        self.encoder = BgeM3Encoder(
            settings.bge_model_path,
            settings.device,
            settings.embedding_batch_size,
            cache_dir=settings.data_root / "embeddings" / "bge-m3",
            model_revision=settings.bge_model_revision,
        )
        self.embed = embed
        if self.embed is None and settings.semantic_enabled and self.encoder.available:
            self.embed = self._dense_embed
        self.milvus = MilvusStore(
            settings.milvus_uri,
            settings.milvus_token.get_secret_value() if settings.milvus_token else None,
        )
        self.entity_extractor = EntityExtractor.from_yaml(
            settings.source_manifest.parent / "security_entities.yaml"
        )

    def ingest_path(self, path: Path) -> IngestionResult:
        document = self.discovery.inspect(path)
        self.store.upsert_document(document, active=False)
        elements = self._parse(document)
        if not elements:
            raise ValueError(f"Document produced no indexable elements: {document.relative_path}")
        self.store.replace_elements(document.source_id, document.version_id, elements)
        symbols = self._symbols(document, elements)
        self.store.replace_code_symbols(document.source_id, document.version_id, symbols)
        warning_list = _quarantine_warnings(elements)
        try:
            chunks = self.chunker.chunk(
                document,
                elements,
                self.embed if self.settings.semantic_enabled else None,
            )
        except (RuntimeError, FileNotFoundError, MemoryError, OSError, ValueError) as error:
            if self.embed is None:
                raise
            warning_list.append(f"语义切片降级：{type(error).__name__}")
            chunks = self.chunker.chunk(document, elements, None)
        if not chunks:
            raise ValueError(f"Document produced no chunks: {document.relative_path}")
        self.store.replace_chunks(document.source_id, document.version_id, chunks)

        candidate_chunks = (*self.store.active_chunks(excluding_source=document.source_id), *chunks)
        candidate_symbols = (
            *self.store.active_code_symbols(excluding_source=document.source_id),
            *symbols,
        )
        tfidf = TfidfIndex()
        tfidf_version = tfidf.fit(candidate_chunks)
        tfidf.save(self.settings.data_root / "tfidf")

        dense = None
        sparse = None
        if self.settings.semantic_enabled:
            if not self.encoder.available:
                warning_list.append("BGE-M3 未安装，跳过语义边与向量写入")
            else:
                try:
                    dense, sparse = self.encoder.encode(
                        [chunk.text for chunk in candidate_chunks]
                    )
                except (RuntimeError, FileNotFoundError, MemoryError, OSError, ValueError) as error:
                    warning_list.append(f"BGE-M3 编码降级：{type(error).__name__}")

        graph = KnowledgeGraph(self.entity_extractor)
        graph.build(candidate_chunks, candidate_symbols)
        if dense is not None:
            graph.add_semantic_edges(
                [chunk.chunk_id for chunk in candidate_chunks],
                dense,
                top_k=self.settings.semantic_edge_top_k,
                threshold=self.settings.semantic_edge_threshold,
                block_size=self.settings.semantic_edge_block_size,
            )
        graph_path = self.settings.data_root / "graph" / f"{tfidf_version}.json.gz"
        graph_version = graph.save(graph_path)
        final_graph_path = self.settings.data_root / "graph" / f"{graph_version}.json.gz"
        if final_graph_path != graph_path:
            graph_path.replace(final_graph_path)

        vectors_written = False
        if dense is not None and sparse is not None:
            if not self.milvus.healthy():
                warning_list.append("Milvus 不可用，跳过向量写入")
            else:
                try:
                    current_start = len(candidate_chunks) - len(chunks)
                    self.milvus.upsert(
                        chunks,
                        dense[current_start:],
                        sparse[current_start:],
                    )
                    vectors_written = True
                except (RuntimeError, FileNotFoundError, MemoryError, OSError) as error:
                    warning_list.append(f"向量写入降级：{type(error).__name__}")

        self.store.publish_version(
            document.source_id,
            document.version_id,
            tfidf_version,
            graph_version,
        )

        if vectors_written:
            try:
                self.milvus.delete_source_versions_except(
                    document.source_id,
                    document.version_id,
                )
            except Exception as error:
                warning_list.append(f"旧向量清理降级：{type(error).__name__}")

        if self.settings.infrastructure_enabled:
            neo4j = Neo4jStore(
                self.settings.neo4j_uri,
                self.settings.neo4j_user,
                self.settings.neo4j_password.get_secret_value(),
            )
            try:
                if not neo4j.healthy():
                    warning_list.append("Neo4j 不可用，保留 NetworkX 图快照")
                else:
                    neo4j.replace_graph(graph.graph, graph_version)
            except Exception as error:
                warning_list.append(f"Neo4j 图同步降级：{type(error).__name__}")
            finally:
                neo4j.close()

        return IngestionResult(
            document,
            elements,
            symbols,
            chunks,
            tfidf_version,
            graph_version,
            tuple(warning_list),
        )

    def ingest_all(self) -> tuple[IngestionResult, ...]:
        results: list[IngestionResult] = []
        for document in self.discovery.discover():
            results.append(self.ingest_path(Path(document.absolute_path)))
        return tuple(results)

    def _parse(self, document: SourceDocument) -> tuple[ParsedElement, ...]:
        if document.format is DocumentFormat.PDF:
            ocr = select_ocr_engine(
                self.settings.ocr_backend,
                languages=self.settings.ocr_languages,
                executable=self.settings.tesseract_executable,
                tessdata_dir=self.settings.tesseract_tessdata_dir,
            )
            extractor = PdfExtractor(
                ocr,
                self.settings.ocr_dpi,
                media_dir=self.settings.corpus_root / ".secknow-media",
            )
            return extractor.to_elements(document, extractor.extract(document))
        if document.format is DocumentFormat.DOCX:
            return self._prefer_unstructured(document, parse_docx)
        if document.format is DocumentFormat.MARKDOWN:
            return self._prefer_unstructured(document, parse_markdown)
        if document.format in {DocumentFormat.TEXT, DocumentFormat.PYTHON}:
            return parse_plain(document)
        raise ValueError(f"Unsupported document format: {document.format}")

    @staticmethod
    def _prefer_unstructured(
        document: SourceDocument,
        fallback: Callable[[SourceDocument], tuple[ParsedElement, ...]],
    ) -> tuple[ParsedElement, ...]:
        try:
            parsed = parse_with_unstructured(document)
            return parsed or fallback(document)
        except (RuntimeError, OSError, ValueError):
            return fallback(document)

    def _symbols(
        self,
        document: SourceDocument,
        elements: Sequence[ParsedElement],
    ) -> tuple[CodeSymbol, ...]:
        symbols: list[CodeSymbol] = []
        diagnostics: list[str] = []
        if document.format is DocumentFormat.PYTHON and elements:
            result = self.code_parser.parse_python(elements[0].element_id, elements[0].text)
            symbols.extend(result.symbols)
            diagnostics.extend(result.diagnostics)
        for element in elements:
            for language, code in self.code_parser.fenced_blocks(element.text):
                if language in {"python", "py"}:
                    result = self.code_parser.parse_python(element.element_id, code)
                elif language not in {"", "text", "plain"}:
                    result = self.code_parser.parse_tree_sitter(element.element_id, language, code)
                else:
                    continue
                symbols.extend(result.symbols)
                diagnostics.extend(result.diagnostics)
        return tuple(symbols)

    def _dense_embed(self, texts: Sequence[str]) -> np.ndarray:
        dense, _ = self.encoder.encode(texts)
        return dense
