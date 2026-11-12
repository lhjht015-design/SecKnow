from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from secknow.domain.ids import (
    normalize_code_text,
    normalize_text,
    sha256_bytes,
    stable_id,
)
from secknow.domain.models import Chunk, Citation, ElementKind, ParsedElement, SourceDocument

SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+(?=[A-Z0-9])|\n{2,}")
SENTENCE_END = re.compile(r"[。！？!?；;\n]")
CLAUSE_END = re.compile(r"[，、,：:]")
TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_./:+-]*|[㐀-鿿]|\d+(?:\.\d+)*")

EmbeddingFunction = Callable[[Sequence[str]], np.ndarray]

CODE_KINDS = {ElementKind.CODE, ElementKind.TABLE}


@dataclass(frozen=True)
class _Unit:
    """最小打包单位：一句话、一行代码或一个表格片段，始终携带来源元素。"""

    text: str
    section: tuple[str, ...]
    elements: tuple[ParsedElement, ...]
    kind: ElementKind

    @property
    def first_element_id(self) -> str:
        return self.elements[0].element_id

    @property
    def last_element_id(self) -> str:
        return self.elements[-1].element_id


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
        units: list[_Unit] = []
        paragraph_buffer: list[ParsedElement] = []

        def flush_paragraphs() -> None:
            # 同一小节的连续段落合流后再切句：PDF 解析常把一个句子拆到多个
            # 元素里，按元素切块会把块边界落在句子中间。
            if not paragraph_buffer:
                return
            units.extend(self._paragraph_units(paragraph_buffer))
            paragraph_buffer.clear()

        for element in elements:
            if element.metadata.get("disposition") in {"ocr_required", "quarantine"}:
                continue
            if not element.text.strip():
                continue
            if element.kind == ElementKind.PARAGRAPH:
                if paragraph_buffer and paragraph_buffer[-1].section_path != element.section_path:
                    flush_paragraphs()
                paragraph_buffer.append(element)
                continue
            flush_paragraphs()
            section = element.section_path
            if element.kind == ElementKind.CODE:
                pieces = self._split_code(element.text)
            elif element.kind == ElementKind.TABLE:
                pieces = self._hard_split(element.text)
            else:
                pieces = self._sentences(element.text)
            for piece in pieces:
                normalized = (
                    normalize_code_text(piece)
                    if element.kind in CODE_KINDS
                    else normalize_text(piece)
                )
                if normalized:
                    units.append(_Unit(normalized, section, (element,), element.kind))
        flush_paragraphs()

        grouped = self._group_units(units, embed)
        chunks: list[Chunk] = []
        for order, group in enumerate(grouped):
            text = self._join_units(group)
            section_path = group[0].section
            source_elements = tuple(
                {element.element_id: element
                 for element in (item for unit in group for item in unit.elements)}.values()
            )
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

    def _group_units(
        self,
        units: Sequence[_Unit],
        embed: EmbeddingFunction | None,
    ) -> list[list[_Unit]]:
        if not units:
            return []
        if embed is not None and len(units) >= 2:
            return self._group_by_embedding(units, embed)
        groups: list[list[_Unit]] = []
        current: list[_Unit] = []
        current_tokens = 0
        for unit in units:
            if current and unit.section != current[-1].section:
                groups.append(current)
                current = []
                current_tokens = 0
            count = self.token_count(unit.text)
            if count > self.chunk_size:
                if current:
                    groups.append(current)
                    current = []
                    current_tokens = 0
                if unit.kind in CODE_KINDS:
                    groups.append([unit])
                else:
                    groups.extend(
                        [_Unit(piece, unit.section, unit.elements, unit.kind)]
                        for piece in self._split_long_sentence(unit.text)
                    )
                continue
            if current and current_tokens + count > self.chunk_size:
                groups.append(current)
                current = self._overlap_units(current)
                current_tokens = sum(self.token_count(item.text) for item in current)
            current.append(unit)
            current_tokens += count
        if current:
            groups.append(current)
        return groups

    def _group_by_embedding(
        self,
        units: Sequence[_Unit],
        embed: EmbeddingFunction,
    ) -> list[list[_Unit]]:
        vectors = np.asarray(embed([unit.text for unit in units]), dtype=np.float32)
        if vectors.shape[0] != len(units):
            raise ValueError("Embedding function returned the wrong number of vectors")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        groups: list[list[_Unit]] = [[units[0]]]
        for index in range(1, len(units)):
            similarity = float(vectors[index - 1] @ vectors[index])
            current_tokens = sum(self.token_count(item.text) for item in groups[-1])
            unit_tokens = self.token_count(units[index].text)
            if (
                units[index].section != groups[-1][-1].section
                or similarity < self.similarity_threshold
                or current_tokens + unit_tokens > self.chunk_size
            ):
                groups.append([units[index]])
            else:
                groups[-1].append(units[index])
        return groups

    def _overlap_units(self, units: Sequence[_Unit]) -> list[_Unit]:
        overlap: list[_Unit] = []
        tokens = 0
        for unit in reversed(units):
            count = self.token_count(unit.text)
            if tokens + count > self.chunk_overlap:
                break
            overlap.insert(0, unit)
            tokens += count
        return overlap

    @staticmethod
    def _join_units(units: Sequence[_Unit]) -> str:
        parts: list[str] = []
        previous: _Unit | None = None
        for unit in units:
            if previous is not None:
                previous_ids = {item.element_id for item in previous.elements}
                current_ids = {item.element_id for item in unit.elements}
                if not previous_ids & current_ids:
                    parts.append("\n\n")
                elif unit.kind in CODE_KINDS or previous.kind in CODE_KINDS:
                    parts.append("\n")
                else:
                    parts.append(" ")
            parts.append(unit.text)
            previous = unit
        return "".join(parts)

    def _paragraph_units(self, elements: Sequence[ParsedElement]) -> list[_Unit]:
        """把同一小节的连续段落拼成一条文本流再切句。"""
        parts: list[str] = []
        spans: list[tuple[int, int, ParsedElement]] = []
        cursor = 0
        for element in elements:
            text = normalize_text(element.text)
            if not text:
                continue
            if parts:
                cursor += 1  # 元素之间只留一个换行，不触发段落边界
            parts.append(text)
            spans.append((cursor, cursor + len(text), element))
            cursor += len(text)
        if not parts:
            return []
        combined = "\n".join(parts)
        section = elements[0].section_path
        units: list[_Unit] = []
        search_from = 0
        span_index = 0
        for sentence in self._sentences(combined):
            start = combined.find(sentence, search_from)
            if start < 0:
                start = search_from
            end = start + len(sentence)
            search_from = end
            while span_index < len(spans) and spans[span_index][1] <= start:
                span_index += 1
            owners: list[ParsedElement] = []
            probe = span_index
            while probe < len(spans) and spans[probe][0] < end:
                owners.append(spans[probe][2])
                probe += 1
            if not owners:
                owners.append(spans[min(span_index, len(spans) - 1)][2])
            units.append(_Unit(sentence, section, tuple(owners), ElementKind.PARAGRAPH))
        return units


    def _sentences(self, text: str) -> list[str]:
        # 只用标点/段落边界切句。llama 的 SentenceSplitter 实际按 token 窗口切，
        # 会把句子从中间截断——这是句中断块的主要来源。
        normalized = normalize_text(text)
        return [piece.strip() for piece in SENTENCE_BOUNDARY.split(normalized)
                if piece.strip()]

    def _split_code(self, text: str) -> list[str]:
        """代码块按行切分；若带围栏，每一片都补上围栏，避免块边界切断代码。"""
        lines = text.splitlines()
        if len(lines) >= 2:
            opening = lines[0]
            match = re.match(r"^(```+|~~~+)", opening.strip())
            if match and lines[-1].strip().startswith(match.group(1)):
                closing = lines[-1]
                inner = lines[1:-1]
                return [
                    f"{opening}\n{piece}\n{closing}"
                    for piece in self._pack_lines(inner)
                ]
        return self._hard_split(text)

    def _pack_lines(self, lines: Sequence[str]) -> list[str]:
        pieces: list[str] = []
        current: list[str] = []
        count = 0
        for line in lines:
            line_count = max(1, self.token_count(line))
            if current and count + line_count > self.chunk_size:
                pieces.append("\n".join(current))
                current = []
                count = 0
            current.append(line)
            count += line_count
        if current:
            pieces.append("\n".join(current))
        return pieces or [""]

    def _split_long_sentence(self, sentence: str) -> list[str]:
        """超长句子在标点处切分：先找句末，再找逗号/冒号，最后退到空白。"""
        tokens = list(TOKEN_PATTERN.finditer(sentence))
        if len(tokens) <= self.chunk_size:
            return [sentence]
        pieces: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            if end >= len(tokens):
                piece = sentence[tokens[start].start():].strip()
                if piece:
                    pieces.append(piece)
                break
            cut = tokens[end - 1].end()
            window_start = tokens[max(start, end - max(8, self.chunk_size // 5))].start()
            candidate = None
            for match in SENTENCE_END.finditer(sentence, window_start, cut):
                candidate = match.end()
            if candidate is None:
                for match in CLAUSE_END.finditer(sentence, window_start, cut):
                    candidate = match.end()
            if candidate is None:
                space = sentence.rfind(" ", tokens[start].start(), cut)
                candidate = space if space > tokens[start].start() else cut
            piece = sentence[tokens[start].start():candidate].strip()
            if piece:
                pieces.append(piece)
            new_start = start
            while new_start < len(tokens) and tokens[new_start].start() < candidate:
                new_start += 1
            start = max(new_start, start + 1)
        return pieces

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

    @staticmethod
    def token_count(text: str) -> int:
        return len(TOKEN_PATTERN.findall(text))
