"""源码入口识别：规则匹配 sink 形态，不做语义理解。"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PRIORITY = {
    "command_injection": 0,
    "format_string": 1,
    "unsafe_copy": 2,
    "blackbox_check": 3,
}


@dataclass(frozen=True)
class SourceEntry:
    category: str
    technique: str
    line: int
    text: str
    confidence: str


_UNSAFE = re.compile(r"\b(gets|strcpy|strcat|sprintf)\s*\(")
_SHELL = re.compile(
    r"\bos\.system\s*\("
    r"|\bsystem\s*\(\s*[A-Za-z_]\w*\s*\)"
    r"|\bsubprocess\.[a-z_]+\([^)]*shell\s*=\s*True"
)
_PRINTF = re.compile(r"\bprintf\s*\(\s*[A-Za-z_]\w*\s*\)")
_READ = re.compile(r"\b(read|fread)\s*\(\s*0\b|\bmemcmp\s*\(")


def detect_source_entries(source: str) -> list[SourceEntry]:
    entries: list[SourceEntry] = []
    seen: set[str] = set()
    lines = source.splitlines()
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "#", "*")):
            continue
        if "command_injection" not in seen and _SHELL.search(stripped):
            seen.add("command_injection")
            entries.append(SourceEntry("command_injection", "命令注入", number,
                                       stripped[:200], "high"))
        if "format_string" not in seen and _PRINTF.search(stripped):
            seen.add("format_string")
            entries.append(SourceEntry("format_string", "格式化字符串", number,
                                       stripped[:200], "medium"))
        if "unsafe_copy" not in seen and _UNSAFE.search(stripped):
            seen.add("unsafe_copy")
            entries.append(SourceEntry("unsafe_copy", "栈溢出 / 无边界拷贝", number,
                                       stripped[:200], "high"))
        if "blackbox_check" not in seen and _READ.search(stripped):
            window = "\n".join(lines[max(0, number - 3): number + 12])
            if "memcmp" in window and ("^" in window or "perm" in window or "table" in window):
                seen.add("blackbox_check")
                entries.append(SourceEntry("blackbox_check", "黑盒变换 / 校验器", number,
                                           stripped[:200], "medium"))
    entries.sort(key=lambda entry: _PRIORITY[entry.category])
    return entries


_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "catch", "with", "print"}


def extract_identifiers(source: str, limit: int = 10) -> list[str]:
    names: list[str] = []
    for match in _CALL.finditer(source):
        name = match.group(1)
        if name in _KEYWORDS or name in names:
            continue
        names.append(name)
        if len(names) >= limit:
            break
    return names
