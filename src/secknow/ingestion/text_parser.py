from __future__ import annotations

import re
from pathlib import Path

from charset_normalizer import from_bytes
from docx import Document as DocxDocument

from secknow.domain.ids import balance_fences, normalize_code_text, normalize_text, stable_id
from secknow.domain.models import ElementKind, ParsedElement, SourceDocument

HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
FENCE = re.compile(r"^\s*```([A-Za-z0-9_+.-]*)\s*$")


def decode_text(path: Path) -> str:
    payload = path.read_bytes()
    match = from_bytes(payload).best()
    if match is None:
        return payload.decode("utf-8", errors="replace")
    return str(match)


def parse_plain(document: SourceDocument) -> tuple[ParsedElement, ...]:
    raw = decode_text(Path(document.absolute_path))
    text = normalize_code_text(raw) if document.format.value == "py" else normalize_text(raw)
    if not text:
        return ()
    return (
        ParsedElement(
            element_id=stable_id("element", document.source_id, document.version_id, 0),
            source_id=document.source_id,
            source_version=document.version_id,
            kind=ElementKind.CODE if document.format.value == "py" else ElementKind.PARAGRAPH,
            text=text,
            order=0,
            parser="charset-normalizer",
        ),
    )


def parse_markdown(document: SourceDocument) -> tuple[ParsedElement, ...]:
    lines = balance_fences(decode_text(Path(document.absolute_path))).splitlines()
    elements: list[ParsedElement] = []
    section_stack: list[str] = []
    buffer: list[str] = []
    buffer_kind = ElementKind.PARAGRAPH
    in_fence = False

    def flush() -> None:
        nonlocal buffer
        joined = "\n".join(buffer)
        text = (normalize_code_text(balance_fences(joined))
                if buffer_kind == ElementKind.CODE
                else normalize_text(joined))
        if text:
            order = len(elements)
            elements.append(
                ParsedElement(
                    element_id=stable_id(
                        "element", document.source_id, document.version_id, order, text[:128]
                    ),
                    source_id=document.source_id,
                    source_version=document.version_id,
                    kind=buffer_kind,
                    text=text,
                    order=order,
                    section_path=tuple(section_stack),
                    parser="markdown-structural",
                )
            )
        buffer = []

    for line in lines:
        heading = HEADING.match(line)
        fence = FENCE.match(line)
        if fence:
            if not in_fence:
                flush()
                in_fence = True
                buffer_kind = ElementKind.CODE
                buffer.append(line)
            else:
                buffer.append(line)
                flush()
                in_fence = False
                buffer_kind = ElementKind.PARAGRAPH
            continue
        if heading and not in_fence:
            flush()
            level = len(heading.group(1))
            section_stack[level - 1 :] = [heading.group(2).strip()]
            order = len(elements)
            elements.append(
                ParsedElement(
                    element_id=stable_id(
                        "element", document.source_id, document.version_id, order, line
                    ),
                    source_id=document.source_id,
                    source_version=document.version_id,
                    kind=ElementKind.HEADING,
                    text=heading.group(2).strip(),
                    order=order,
                    section_path=tuple(section_stack),
                    parser="markdown-structural",
                )
            )
            continue
        if not line.strip() and not in_fence:
            flush()
        else:
            buffer.append(line)
    flush()
    return tuple(elements)


def parse_docx(document: SourceDocument) -> tuple[ParsedElement, ...]:
    package = DocxDocument(document.absolute_path)
    elements: list[ParsedElement] = []
    section_stack: list[str] = []
    for paragraph in package.paragraphs:
        text = normalize_text(paragraph.text)
        if not text:
            continue
        style = paragraph.style.name.casefold() if paragraph.style else ""
        kind = ElementKind.HEADING if style.startswith("heading") else ElementKind.PARAGRAPH
        if kind is ElementKind.HEADING:
            match = re.search(r"(\d+)$", style)
            level = int(match.group(1)) if match else 1
            section_stack[level - 1 :] = [text]
        order = len(elements)
        elements.append(
            ParsedElement(
                element_id=stable_id(
                    "element", document.source_id, document.version_id, order, text[:128]
                ),
                source_id=document.source_id,
                source_version=document.version_id,
                kind=kind,
                text=text,
                order=order,
                section_path=tuple(section_stack),
                parser="python-docx",
            )
        )
    for table in package.tables:
        rows = ["\t".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        text = normalize_text("\n".join(rows))
        if text:
            order = len(elements)
            elements.append(
                ParsedElement(
                    element_id=stable_id(
                        "element", document.source_id, document.version_id, order, text[:128]
                    ),
                    source_id=document.source_id,
                    source_version=document.version_id,
                    kind=ElementKind.TABLE,
                    text=text,
                    order=order,
                    section_path=tuple(section_stack),
                    parser="python-docx",
                )
            )
    return tuple(elements)
