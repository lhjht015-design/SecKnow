from __future__ import annotations

"""代码文件专用清洗（在 basic + `normalize_code_whitespace` 之后执行）。

**处理内容**
- 每行行尾空白（`rstrip`）：清除 diff 噪声，**保留行首缩进**（不把整行 trim 掉左侧空格）。
- 全文首尾 `strip()`：去掉文件头尾多余空行；不会去掉首行代码的缩进
  （按行 `rstrip` 后再 `join`，非空首行的前导空格保留）。

**与 `normalize.py` 的配合**
- `normalize_for_file_type(..., file_type="code")` 已做：宽空格→ASCII 空格、NFC、行内连续空格压缩。
- 本函数不再做全角 ASCII 标点映射，避免改动字符串字面量中的刻意 Unicode。

**刻意不做**
- 不做 black/ruff 等格式化、不删注释、不改引号风格。
"""


def clean_code(text: str) -> str:
    """去除行尾空格，保留缩进与换行结构。"""
    if not text:
        return ""
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()
