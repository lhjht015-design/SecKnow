from __future__ import annotations

"""Markdown 专用清洗（在 basic + 全局标点规范化之后执行）。

**处理内容**
- 去掉 HTML 注释 `<!-- ... -->`（含跨行），减少从 MD 导出或模板残留噪声。
- 行尾空白清理、将 3 行及以上连续空行压成 2 行，避免 chunk 被无意义空白撑大。

**刻意不做**
- 不解析 AST、不删除合法 Markdown 语法（标题/列表/代码围栏等），避免破坏原文结构。
- 不做链接自动规范化；复杂场景建议专用 Markdown linter 在入库前处理。
"""

import re

# 跨行 HTML 注释（常见于模板或编辑器元数据）
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_MULTI_BLANK = re.compile(r"\n{3,}")


def clean_markdown(text: str) -> str:
    """Markdown 轻量清洗：去 HTML 注释、行尾空白、压缩过多空行。"""
    if not text:
        return ""
    text = _HTML_COMMENT.sub("", text)
    lines = [line.rstrip() for line in text.split("\n")]
    body = "\n".join(lines).strip()
    return _MULTI_BLANK.sub("\n\n", body)
