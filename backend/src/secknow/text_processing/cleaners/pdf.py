from __future__ import annotations

"""PDF 文本清洗（页眉页脚弱化）。

**设计背景**
- 旧版 `load_pdf` 用单换行拼接各页，清洗层无法区分「页边界」，也就无法做跨页统计。
- 现由 `loaders/pdf_loader` 在页与页之间插入本模块定义的 `PDF_PAGE_BOUNDARY` 常量；
  本文件在 `clean_document_text(..., file_type="pdf")` 中消费该标记。

**页眉页脚策略（启发式，非 OCR 版面分析）**
- 对「非空页」分别取首条非空行、末条非空行，在全文范围内统计出现次数。
- 若某一行作为「多页的首行」出现比例 ≥ 阈值，且长度在合理范围内，则视为重复页眉，从各页删去一次该行的首次匹配。
- 末行同理视为页脚。
- 阈值、比例、最大行宽可通过环境变量调整（见下方常量）。

**误伤说明**
- 封面/扉页与正文首行相同时可能多删一行；学术论文标题重复出现于每页时也可能被误判。
- 若文本中不含 `PDF_PAGE_BOUNDARY`（例如手工拼装的字符串），本函数直接返回原文，不做页级处理。

**与 OCR 的关系**
- `loaders/pdf_advanced` 可能对扫描页做 Tesseract OCR：断行、空格与嵌入层不一致，
  页眉统计仍基于「整行相同」启发式，若 OCR 把页眉拆碎，重复行检测效果会下降，可依赖
  `PDF_HEADER_*` 环境变量调参。
"""

import math
import os
from collections import Counter

# 必须整行匹配且不含易被 basic_clean 误删的字符；与 loaders/pdf_loader 保持一致。
PDF_PAGE_BOUNDARY = "\n__SECKNOW_PAGE_BOUNDARY__\n"

# 页眉行至少多长才参与统计（过滤页码「1」等极短噪声时可调大）
_DEFAULT_MIN_LINE_LEN = int(os.getenv("PDF_HEADER_MIN_LINE_LEN", "3"))
# 页眉行超过该长度不视为页眉（避免误删长标题）
_DEFAULT_MAX_LINE_LEN = int(os.getenv("PDF_HEADER_MAX_LINE_LEN", "200"))
# 非空页中，首行相同比例 ≥ 该值则触发删除（0~1）
_DEFAULT_EDGE_REPEAT_RATIO = float(os.getenv("PDF_HEADER_REPEAT_RATIO", "0.65"))


def _first_non_empty_line(text: str) -> str:
    for line in text.split("\n"):
        s = line.strip()
        if s:
            return s
    return ""


def _last_non_empty_line(text: str) -> str:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return lines[-1] if lines else ""


def _remove_first_line_matching(text: str, target: str) -> str:
    """删除首次出现的、strip 后与 target 相等的整行（含该行换行）。"""
    if not text.strip() or not target:
        return text
    lines = text.split("\n")
    out: list[str] = []
    removed = False
    for line in lines:
        if not removed and line.strip() == target:
            removed = True
            continue
        out.append(line)
    return "\n".join(out)


def _remove_last_line_matching(text: str, target: str) -> str:
    """删除最后一次出现的、strip 后与 target 相等的整行。"""
    if not text.strip() or not target:
        return text
    lines = text.split("\n")
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip() == target:
            del lines[idx]
            break
    return "\n".join(lines)


def clean_pdf_document(
    text: str,
    *,
    min_line_len: int = _DEFAULT_MIN_LINE_LEN,
    max_line_len: int = _DEFAULT_MAX_LINE_LEN,
    repeat_ratio: float = _DEFAULT_EDGE_REPEAT_RATIO,
) -> str:
    """识别 `PDF_PAGE_BOUNDARY`，弱化重复页眉页脚后用双换行拼接各页正文。"""
    repeat_ratio = min(1.0, max(0.0, repeat_ratio))
    if PDF_PAGE_BOUNDARY not in text:
        return text

    raw_pages = text.split(PDF_PAGE_BOUNDARY)
    pages = [p.strip() for p in raw_pages]

    non_empty = [p for p in pages if p.strip()]
    if len(non_empty) < 2:
        return "\n\n".join(non_empty)

    threshold = max(2, math.ceil(repeat_ratio * len(non_empty)))

    def maybe_strip_edge(
        pages_in: list[str], *, edge: str
    ) -> list[str]:
        if edge == "first":
            lines = [_first_non_empty_line(p) for p in pages_in if p.strip()]
        else:
            lines = [_last_non_empty_line(p) for p in pages_in if p.strip()]
        if not lines:
            return pages_in
        common, cnt = Counter(lines).most_common(1)[0]
        if (
            cnt < threshold
            or not (min_line_len <= len(common) <= max_line_len)
        ):
            return pages_in
        if edge == "first":
            return [_remove_first_line_matching(p, common) for p in pages_in]
        return [_remove_last_line_matching(p, common) for p in pages_in]

    pages = maybe_strip_edge(pages, edge="first")
    pages = maybe_strip_edge(pages, edge="last")

    merged = [p.strip() for p in pages if p.strip()]
    return "\n\n".join(merged)
