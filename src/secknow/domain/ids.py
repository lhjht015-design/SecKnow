from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Any

NAMESPACE = uuid.UUID("f7e26554-1cad-42af-b37d-7bc4b01275b1")
FENCE_LINE = re.compile(r"^(\s*)(```+|~~~+)([^`~]*)$")


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


def line_fence_open(value: str) -> bool:
    """块尾是否仍处于未闭合围栏内。

    判定与前端 `splitEvidence` 一致：一行去掉行首空白后以 ``` 开头即为围栏，
    因此行内出现的 ```（如“页码。 ```”）不算围栏。
    """
    in_fence = False
    for line in value.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
    return in_fence


def balance_fences(value: str) -> str:
    """把代码围栏规范成成对的开/闭行。

    - 同时识别 ``` 与 ~~~；输出统一成 ```；
    - 围栏允许缩进（Markdown 列表里的代码块常缩进 4 个空格）；
    - 开围栏后紧跟的重复开围栏（作者/生成器写成 `~~~python` + ` ```python `）
      直接丢弃，避免整篇围栏奇偶翻转；
    - 行内闭合（代码行末尾直接跟 ```）拆成代码行 + 独立的闭合围栏；
    - 文本结束时仍未闭合，补一行闭合围栏。
    """
    output: list[str] = []
    open_marker = ""
    open_indent = ""
    seen_content = False
    for line in value.splitlines():
        match = FENCE_LINE.match(line)
        if not open_marker:
            if match:
                open_indent = match.group(1)
                open_marker = match.group(2)
                output.append(f"{open_indent}```{match.group(3).strip()}")
                seen_content = False
            else:
                output.append(line)
            continue
        if match and not seen_content and match.group(3).strip():
            # 开围栏后的重复开围栏，丢掉第二个
            continue
        if match and not match.group(3).strip():
            output.append(f"{open_indent}```")
            open_marker = ""
            continue
        stripped = line.strip()
        if "```" in stripped and stripped.endswith("```"):
            before = line.rstrip()[:-3].rstrip()
            if before.strip():
                output.append(before)
                seen_content = True
            output.append(f"{open_indent}```")
            open_marker = ""
            continue
        if stripped:
            seen_content = True
        output.append(line)
    if open_marker:
        output.append(f"{open_indent}```")
    return "\n".join(output)


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
