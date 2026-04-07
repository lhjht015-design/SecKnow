from __future__ import annotations

"""基础文本清洗逻辑（全类型通用第一步）。

**职责边界（给接手同事的快速说明）**
1. **换行规范化**：`\\r\\n`、`\\r` 一律转为 `\\n`，避免同一文档混用多种换行符。
2. **控制字符**：删除 C0 控制区中除 `\\t`、`\\n` 外的字符（含 `\\x00`、DEL 等），
   防止 PDF/Office 导出带来的不可见噪声；**不**删除 `\\t`，以便表格/缩进仍可用。
3. **空行剔除**：按行 `strip` 后丢弃纯空白行，再用单个 `\\n` 连接——
   这会**抹掉「仅用于视觉分隔的空行」**；段落语义主要交给后续分块策略。
4. **BOM**：去掉 UTF-8 文件头 `\\ufeff`（ZWNBSP），避免首字符污染 `content_hash` 与展示。

**刻意不做的事（由其他模块负责）**
- 全角/半角标点、全角空格 → 见 `normalize.py`
- 页眉页脚 → 见 `pdf.py`（依赖 loader 插入页界）
- Markdown/代码语义 → 见 `markdown.py` / `code.py`
"""

import re

# 保留制表符(0x09)与换行(0x0A)，其余 C0 控制字符及 DEL 删除。
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def basic_clean(text: str) -> str:
    """执行通用机械清洗：BOM、换行统一、控制字符、剔除空白行。"""
    if not text:
        return ""

    if text.startswith("\ufeff"):
        text = text[1:]

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _CONTROL_CHAR_RE.sub("", normalized)

    cleaned_lines: list[str] = []
    for line in normalized.split("\n"):
        stripped = line.strip()
        if stripped:
            cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines).strip()
