from __future__ import annotations

"""中英文标点与 Unicode 空白规范化。

**与 `basic_clean` 的分工**
- `basic_clean`：ASCII 控制字符、\\r/\\n 统一、剔除纯空行（机械清洗）。
- 本模块：面向「可读文本」的标点与空白兼容（全角 ASCII 区、全角空格等）。
  不重复删除 C0 控制符，假定调用方已先跑过 `basic_clean`。

**全角半角（FF01–FF5E）**
- Unicode 全角 ASCII 区（FULLWIDTH EXCLAMATION MARK 等）映射到 U+0021–U+007E。
- 中文专用标点（如 U+3002 「。」、U+3001 「、」）**不在**该区间内，不会被改写。
- 全角逗号 U+FF0C 会变为英文逗号 U+002C——多数检索与向量化场景可接受；若业务强依赖显示形态，可后续加开关。

**代码文件（for_code=True）**
- 不对 FF01–FF5E 做批量映射，避免无意改变源码中刻意使用的全角字符。
- 仅将常见「宽空格」类字符替换为普通 ASCII 空格，降低 diff 噪声。
"""

import re
import unicodedata


# 连续 ASCII 空格/制表（行内）可压成单空格；保留换行结构。
_COLLAPSE_INLINE_SPACES = re.compile(r"[ \t]{2,}")


def map_fullwidth_ascii_block(text: str) -> str:
    """将 U+FF01–U+FF5E 映射到 U+0021–U+007E（标准全角半角对应）。"""
    out: list[str] = []
    for ch in text:
        o = ord(ch)
        if 0xFF01 <= o <= 0xFF5E:
            out.append(chr(o - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def normalize_unicode_spaces(text: str) -> str:
    """将各类「宽空格」统一为 U+0020，便于后续分词与哈希稳定。"""
    # \u3000 全角空格；\u00a0 不换行空格；\u2000-\u200A 一般标点空白；\u202F 窄不换行空格；\u205F 中等数学空格
    result: list[str] = []
    for ch in text:
        o = ord(ch)
        if ch == "\u3000" or ch == "\u00a0" or ch == "\u202f" or ch == "\u205f":
            result.append(" ")
        elif 0x2000 <= o <= 0x200A:
            result.append(" ")
        else:
            result.append(ch)
    return "".join(result)


def normalize_prose_punctuation(text: str) -> str:
    """正文用：宽空格 + 全角 ASCII 区 + NFC 归一化 + 行内多余空格压缩。"""
    if not text:
        return ""
    text = normalize_unicode_spaces(text)
    text = map_fullwidth_ascii_block(text)
    # NFC：组合字符稳定，不启用 NFKC（避免过度兼容分解汉字/符号）
    text = unicodedata.normalize("NFC", text)
    lines = []
    for line in text.split("\n"):
        lines.append(_COLLAPSE_INLINE_SPACES.sub(" ", line).rstrip())
    return "\n".join(lines).strip()


def normalize_code_whitespace(text: str) -> str:
    """源码用：仅处理空白类兼容，不映射全角 ASCII 标点块。"""
    if not text:
        return ""
    text = normalize_unicode_spaces(text)
    text = unicodedata.normalize("NFC", text)
    lines = []
    for line in text.split("\n"):
        lines.append(_COLLAPSE_INLINE_SPACES.sub(" ", line).rstrip())
    return "\n".join(lines).strip()


def normalize_for_file_type(text: str, *, file_type: str) -> str:
    """供 `document_clean` 调用：按类型选择正文/代码策略。"""
    if file_type == "code":
        return normalize_code_whitespace(text)
    return normalize_prose_punctuation(text)


def normalize_text(text: str, *, for_code: bool = False) -> str:
    """历史/通用入口：等价于 prose 或 code 分支。"""
    if for_code:
        return normalize_code_whitespace(text)
    return normalize_prose_punctuation(text)
