from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np

from secknow.domain.ids import normalize_text, sha256_bytes, stable_id
from secknow.domain.models import Chunk, Citation, ElementKind, ParsedElement, SourceDocument

SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+(?=[A-Z0-9])|\n{2,}")
TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_./:+-]*|[㐀-鿿]|\d+(?:\.\d+)*")

EmbeddingFunction = Callable[[Sequence[str]], np.ndarray]


class SemanticChunker:
    def __init__(
        self,
        chunk_size: int = 480,
        chunk_overlap: int = 64,
        similarity_threshold: float = 0.48,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold

    def chunk(
        self,
        document: SourceDocument,
        elements: Sequence[ParsedElement],
        embed: EmbeddingFunction | None = None,
    ) -> tuple[Chunk, ...]:
        drafts: list[tuple[str, tuple[str, ...], tuple[ParsedElement, ...]]] = []
        for element in elements:
            if element.metadata.get("disposition") in {"ocr_required", "quarantine"}:
                continue
            if not element.text.strip():
                continue
            if element.kind in {ElementKind.CODE, ElementKind.TABLE}:
                pieces = self._hard_split(element.text)
            else:
                pieces = self._semantic_pieces(element.text, embed)
            for piece in pieces:
                normalized = normalize_text(piece)
                if normalized:
                    drafts.append((normalized, element.section_path, (element,)))

        merged = self._merge_small(drafts)
        chunks: list[Chunk] = []
        for order, (text, section_path, source_elements) in enumerate(merged):
            first = source_elements[0]
            last = source_elements[-1]
            content_hash = sha256_bytes(text.encode("utf-8"))
            chunk_id = stable_id(
                "chunk",
                document.source_id,
                document.version_id,
                first.element_id,
                last.element_id,
                content_hash,
            )
            pages = [element.page for element in source_elements if element.page is not None]
            boxes = tuple(
                element.bounding_box
                for element in source_elements
                if element.bounding_box is not None
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_id=document.source_id,
                    source_version=document.version_id,
                    text=text,
                    content_sha256=content_hash,
                    element_ids=tuple(element.element_id for element in source_elements),
                    order=order,
                    token_count=max(1, self.token_count(text)),
                    section_path=section_path,
                    citation=Citation(
                        source_id=document.source_id,
                        source_version=document.version_id,
                        title=document.title,
                        relative_path=document.relative_path,
                        page_start=min(pages) if pages else None,
                        page_end=max(pages) if pages else None,
                        section_path=section_path,
                        bounding_boxes=boxes,
                        extraction_confidence=min(
                            element.extraction_confidence for element in source_elements
                        ),
                    ),
                    metadata={
                        "element_kinds": sorted({element.kind.value for element in source_elements})
                    },
                )
            )
        for index, chunk in enumerate(chunks):
            chunks[index] = chunk.model_copy(
                update={
                    "previous_chunk_id": chunks[index - 1].chunk_id if index else None,
                    "next_chunk_id": (
                        chunks[index + 1].chunk_id if index + 1 < len(chunks) else None
                    ),
                }
            )
        return tuple(chunks)

    def _semantic_pieces(
        self,
        text: str,
        embed: EmbeddingFunction | None,
    ) -> list[str]:
        sentences = self._sentences(text)
        if not sentences:
            return []
        if embed is None or len(sentences) < 2:
            return self._pack(sentences)
        vectors = np.asarray(embed(sentences), dtype=np.float32)
        if vectors.shape[0] != len(sentences):
            raise ValueError("Embedding function returned the wrong number of vectors")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        groups: list[list[str]] = [[sentences[0]]]
        for index in range(1, len(sentences)):
            similarity = float(vectors[index - 1] @ vectors[index])
            current_tokens = self.token_count(" ".join(groups[-1]))
            sentence_tokens = self.token_count(sentences[index])
            if (
                similarity < self.similarity_threshold
                or current_tokens + sentence_tokens > self.chunk_size
            ):
                groups.append([sentences[index]])
            else:
                groups[-1].append(sentences[index])
        return [" ".join(group) for group in groups]

    def _sentences(self, text: str) -> list[str]:
        normalized = normalize_text(text)
        try:
            from llama_index.core.node_parser import SentenceSplitter

            splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            pieces = splitter.split_text(normalized)
            if pieces:
                return [piece.strip() for piece in pieces if piece.strip()]
        except ImportError:
            pass
        return [piece.strip() for piece in SENTENCE_BOUNDARY.split(normalized) if piece.strip()]

    def _pack(self, sentences: Sequence[str]) -> list[str]:
        packed: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for sentence in sentences:
            count = self.token_count(sentence)
            if count > self.chunk_size:
                if current:
                    packed.append(" ".join(current))
                    current = []
                    current_tokens = 0
                packed.extend(self._hard_split(sentence))
                continue
            if current and current_tokens + count > self.chunk_size:
                packed.append(" ".join(current))
                overlap: list[str] = []
                overlap_tokens = 0
                for previous in reversed(current):
                    previous_count = self.token_count(previous)
                    if overlap_tokens + previous_count > self.chunk_overlap:
                        break
                    overlap.insert(0, previous)
                    overlap_tokens += previous_count
                current = overlap
                current_tokens = overlap_tokens
            current.append(sentence)
            current_tokens += count
        if current:
            packed.append(" ".join(current))
        return packed

    def _hard_split(self, text: str) -> list[str]:
        tokens = TOKEN_PATTERN.findall(text)
        if len(tokens) <= self.chunk_size:
            return [text]
        lines = text.splitlines(keepends=True)
        pieces: list[str] = []
        current: list[str] = []
        count = 0
        for line in lines:
            line_count = max(1, self.token_count(line))
            if current and count + line_count > self.chunk_size:
                pieces.append("".join(current).strip())
                current = []
                count = 0
            current.append(line)
            count += line_count
        if current:
            pieces.append("".join(current).strip())
        if len(pieces) == 1:
            width = max(1, int(len(text) * self.chunk_size / len(tokens)))
            return [text[index : index + width] for index in range(0, len(text), width)]
        return pieces

    def _merge_small(
        self,
        drafts: list[tuple[str, tuple[str, ...], tuple[ParsedElement, ...]]],
    ) -> list[tuple[str, tuple[str, ...], tuple[ParsedElement, ...]]]:
        if not drafts:
            return []
        merged: list[tuple[str, tuple[str, ...], tuple[ParsedElement, ...]]] = []
        for text, section, elements in drafts:
            if (
                merged
                and merged[-1][1] == section
                and self.token_count(merged[-1][0]) + self.token_count(text) <= self.chunk_size
            ):
                previous_text, _, previous_elements = merged[-1]
                merged[-1] = (
                    f"{previous_text}\n\n{text}",
                    section,
                    tuple(
                        {
                            element.element_id: element
                            for element in (*previous_elements, *elements)
                        }.values()
                    ),
                )
            else:
                merged.append((text, section, elements))
        return merged

    @staticmethod
    def token_count(text: str) -> int:
        return len(TOKEN_PATTERN.findall(text))
