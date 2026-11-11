from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from secknow.config import Settings

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".py", ".md"}


def resolve_corpus_path(
    settings: Settings,
    relative_path: str,
    *,
    require_file: bool | None = None,
) -> Path:
    relative = Path(relative_path.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise HTTPException(status_code=400, detail="path must be corpus-relative")
    root = settings.corpus_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise HTTPException(status_code=400, detail="path escapes the corpus root") from error
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="corpus path not found")
    if require_file is True and not candidate.is_file():
        raise HTTPException(status_code=400, detail="path must identify a file")
    if require_file is False and not candidate.is_dir():
        raise HTTPException(status_code=400, detail="path must identify a directory")
    return candidate


def indexable_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.casefold() in SUPPORTED_EXTENSIONS else []
    iterator = path.rglob("*") if recursive else path.glob("*")
    return sorted(
        item
        for item in iterator
        if item.is_file() and item.suffix.casefold() in SUPPORTED_EXTENSIONS
    )


def parse_json_fields(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    for key in ("payload_json", "metadata_json", "manifest_json"):
        if key in output:
            raw = output.pop(key)
            try:
                output[key.removesuffix("_json")] = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                output[key.removesuffix("_json")] = None
    return output


def media_type_for(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"
