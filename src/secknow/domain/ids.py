from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Any

NAMESPACE = uuid.UUID("f7e26554-1cad-42af-b37d-7bc4b01275b1")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def normalize_code_text(value: str) -> str:
    """保留缩进与行结构的规范化，供代码块与表格使用。

    普通的 normalize_text 会把所有制表符/连续空格压成一个空格，
    代码缩进因此完全丢失（实测 43% 的代码元素有缩进，但块里只剩一列）。
    """
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in value.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    result: list[str] = []
    blanks = 0
    for line in lines:
        if line.strip():
            blanks = 0
            result.append(line)
        else:
            blanks += 1
            if blanks <= 2:
                result.append("")
    return "\n".join(result)


def stable_id(kind: str, *parts: object) -> str:
    normalized = [kind, *(normalize_text(str(part)) for part in parts)]
    return f"{kind}:{uuid.uuid5(NAMESPACE, json.dumps(normalized, ensure_ascii=False))}"


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256_bytes(encoded)
