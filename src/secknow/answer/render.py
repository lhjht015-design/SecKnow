from __future__ import annotations

import html
from typing import cast

import bleach
from markdown_it import MarkdownIt

from secknow.answer.evidence import EvidenceAnswer

ALLOWED_TAGS = {
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "ul",
    "ol",
    "li",
    "strong",
    "em",
    "code",
    "pre",
    "blockquote",
    "hr",
    "br",
}


def evidence_markdown(answer: EvidenceAnswer) -> str:
    lines = [f"# {answer.query}", "", answer.summary]
    for section in answer.sections:
        lines.extend(["", f"## {section.title}"])
        for item in section.items:
            citation = item.citation.title
            if item.citation.page_start:
                citation += f" · 第 {item.citation.page_start} 页"
            lines.extend(
                [
                    "",
                    f"### {citation}",
                    "",
                    item.text,
                    "",
                    f"相关排序分数：`{item.score:.6f}`",
                ]
            )
    if answer.warnings:
        lines.extend(["", "## 运行提示"])
        lines.extend(f"- {warning}" for warning in answer.warnings)
    return "\n".join(lines)


def markdown_to_safe_html(markdown: str) -> str:
    renderer = MarkdownIt("commonmark", {"html": False, "linkify": False})
    rendered = renderer.render(markdown)
    return cast(
        str,
        bleach.clean(
            rendered,
            tags=ALLOWED_TAGS,
            attributes={},
            protocols=set(),
            strip=True,
        ),
    )


def plain_text_excerpt(text: str, limit: int = 320) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return html.escape(normalized[: limit - 1].rstrip()) + "…"
