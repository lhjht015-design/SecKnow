from __future__ import annotations

import mimetypes
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from secknow.domain.ids import sha256_file, stable_id
from secknow.domain.models import DocumentFormat, SourceDocument

SUPPORTED_FORMATS = {
    ".pdf": DocumentFormat.PDF,
    ".docx": DocumentFormat.DOCX,
    ".txt": DocumentFormat.TEXT,
    ".py": DocumentFormat.PYTHON,
    ".md": DocumentFormat.MARKDOWN,
}
MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".py": "text/x-python",
    ".md": "text/markdown",
}
MAGIC = {
    ".pdf": b"%PDF-",
    ".docx": b"PK\x03\x04",
}
PIPELINE_VERSION = "pipeline-v2"


class UnsafeCorpusPath(ValueError):
    pass


class UnsupportedDocument(ValueError):
    pass


class DocumentDiscovery:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)

    def validate_path(self, path: Path) -> Path:
        if path.is_symlink():
            raise UnsafeCorpusPath(f"Symbolic links are not accepted: {path}")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise UnsafeCorpusPath(f"Path leaves corpus root: {path}") from error
        if not resolved.is_file():
            raise UnsafeCorpusPath(f"Not a regular file: {path}")
        return resolved

    def discover(self) -> list[SourceDocument]:
        documents: list[SourceDocument] = []
        for path in self._iter_candidates():
            try:
                documents.append(self.inspect(path))
            except (OSError, UnsupportedDocument, UnsafeCorpusPath):
                continue
        return sorted(documents, key=lambda item: item.relative_path.casefold())

    def _iter_candidates(self) -> Iterator[Path]:
        for path in self.root.rglob("*"):
            if path.is_file() and path.suffix.casefold() in SUPPORTED_FORMATS:
                yield path

    def inspect(self, path: Path) -> SourceDocument:
        resolved = self.validate_path(path)
        suffix = resolved.suffix.casefold()
        if suffix not in SUPPORTED_FORMATS:
            raise UnsupportedDocument(f"Unsupported extension: {suffix}")
        self._verify_magic(resolved, suffix)
        relative = resolved.relative_to(self.root).as_posix()
        digest = sha256_file(resolved)
        stat = resolved.stat()
        source_id = stable_id("document", relative.casefold())
        version_id = stable_id("version", source_id, digest, PIPELINE_VERSION)
        media_type = MEDIA_TYPES.get(suffix) or mimetypes.guess_type(resolved.name)[0]
        return SourceDocument(
            source_id=source_id,
            version_id=version_id,
            title=resolved.stem,
            relative_path=relative,
            absolute_path=str(resolved),
            format=SUPPORTED_FORMATS[suffix],
            media_type=media_type or "application/octet-stream",
            size_bytes=stat.st_size,
            sha256=digest,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            metadata={"pipeline_version": PIPELINE_VERSION},
        )

    @staticmethod
    def _verify_magic(path: Path, suffix: str) -> None:
        expected = MAGIC.get(suffix)
        if expected is None:
            return
        with path.open("rb") as stream:
            header = stream.read(len(expected))
        if header != expected:
            raise UnsupportedDocument(f"File signature does not match {suffix}: {path.name}")

    def llamaindex_documents(self, paths: list[Path]) -> list[object]:
        try:
            from llama_index.core import SimpleDirectoryReader
        except ImportError as error:
            raise RuntimeError("Install SecKnow with the 'parse' extra for LlamaIndex") from error
        safe_paths = [str(self.validate_path(path)) for path in paths]
        return list(SimpleDirectoryReader(input_files=safe_paths).load_data())
