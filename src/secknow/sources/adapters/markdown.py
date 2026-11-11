from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from secknow.domain.models import SourceRecord
from secknow.sources.adapters.base import AdapterContext, BaseSourceAdapter

_HEADING = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_TITLE = re.compile(r"^title:\s*[\"']?(.+?)[\"']?\s*$", re.IGNORECASE | re.MULTILINE)


class GitMarkdownTreeAdapter(BaseSourceAdapter):
    adapter_name = "git_markdown"
    suffixes = (".md", ".markdown")

    def iter_records(self, path: Path, context: AdapterContext) -> Iterator[SourceRecord]:
        root = path if path.is_dir() else path.parent
        for file in self.files(path):
            body = file.read_text(encoding="utf-8-sig")
            heading = _HEADING.search(body)
            frontmatter_title = _TITLE.search(body[:4096]) if body.startswith("---") else None
            title_match = frontmatter_title or heading
            title = title_match.group(1).strip() if title_match else file.stem.replace("-", " ")
            relative = file.relative_to(root).as_posix()
            yield self.record(
                context=context,
                path=file,
                native_id=relative,
                title=title,
                body=body,
                metadata={"relative_path": relative, "document_format": "markdown"},
            )


GenericGitMarkdownTreeAdapter = GitMarkdownTreeAdapter
