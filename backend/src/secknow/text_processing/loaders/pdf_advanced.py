from __future__ import annotations

"""PDF 高级抽取：嵌入文本、表格 Markdown、扫描页 OCR。

================================================================================
一、能力概览（维护者请先读本节）
================================================================================
1. **矢量/电子版 PDF**
   - 使用 PyMuPDF（`fitz`）`page.get_text("text")` 读取嵌入文字层。
2. **扫描版、纯图片页**
   - 当单页嵌入字符数低于阈值（默认见 `PDF_OCR_MIN_TEXT_CHARS`）且全局开启 OCR 时，
     将页面按倍率光栅化（`page.get_pixmap`）后送 **Tesseract** 做整页 OCR。
3. **表格（复杂表格 PDF 的一阶支持）**
   - 若当前 PyMuPDF 版本提供 `page.find_tables()`，则将每个表 `extract()` 为二维数组，
     再转为 **Markdown 管道表**，置于该页文本块**之前**，便于分块时保留行列结构。

================================================================================
二、依赖与部署
================================================================================
- **必选**：`pymupdf`（`import fitz`）。
- **OCR 可选**：系统需安装 `tesseract` 可执行文件；Python 包 `pytesseract` 仅为桥接。
  - Windows：安装 Tesseract 安装程序并将目录加入 PATH，或设置环境变量 `TESSDATA_PREFIX`。
  - 中文扫描件：需安装 `chi_sim`（简体）等语言数据，与 `PDF_OCR_LANG`（默认 `chi_sim+eng`）一致。
- 若未检测到 Tesseract：OCR 路径**静默跳过**，仅返回嵌入层文本（扫描件可能几乎为空）。

================================================================================
三、环境变量（统一在此模块读取，避免与 config.py 循环依赖）
================================================================================
- `PDF_OCR_ENABLED`：是否启用 OCR（默认开启：`1`/`true`）。
- `PDF_OCR_MIN_TEXT_CHARS`：单页嵌入文本长度 < 该值则尝试 OCR（默认 `40`）。
- `PDF_OCR_ZOOM`：渲染矩阵倍率，越大越清晰但越慢（默认 `2.5`，限制在 1~4）。
- `PDF_OCR_LANG`：传给 `tesseract -l`（默认 `chi_sim+eng`）。
- `PDF_TABLES_ENABLED`：是否调用 `find_tables`（默认 `1`）。

================================================================================
四、已知局限（勿向业务承诺「完美还原版式」）
================================================================================
- 多栏、脚注、侧栏：`get_text` 与表格框线检测都可能乱序或串栏。
- 合并单元格、嵌套表：Markdown 表无法表达时只能近似展开或留空。
- OCR：倾斜、低分辨率、艺术字会导致错字；表格线密集时 OCR 与表格抽取可能重复内容。
- 表格块与正文在矢量 PDF 中地理重叠时，可能出现「表 + 文」重复片段，由下游去重/检索消化。

================================================================================
五、与 `cleaners/pdf.py` 的关系
================================================================================
- `loaders/pdf_loader` 在页间插入 `PDF_PAGE_BOUNDARY`，供 `clean_pdf_document` 做页眉页脚统计。
- OCR 后的换行、断词风格与嵌入文本不同，页眉启发式仍适用但阈值需按数据调参。
"""

import io
import os
from typing import Any

import fitz

# ---- 环境读取 ----

def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _ocr_min_chars() -> int:
    raw = os.getenv("PDF_OCR_MIN_TEXT_CHARS", "40").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 40


def _ocr_zoom() -> float:
    raw = os.getenv("PDF_OCR_ZOOM", "2.5").strip()
    try:
        z = float(raw)
    except ValueError:
        return 2.5
    return max(1.0, min(z, 4.0))


def _ocr_lang() -> str:
    return os.getenv("PDF_OCR_LANG", "chi_sim+eng").strip() or "eng"


def _tables_enabled() -> bool:
    return _env_flag("PDF_TABLES_ENABLED", "1")


def _ocr_enabled() -> bool:
    return _env_flag("PDF_OCR_ENABLED", "1")


# ---- Tesseract 可用性（懒探测）----
_tesseract_checked: bool = False
_tesseract_usable: bool = False


def _tesseract_available() -> bool:
    """首次需要 OCR 时探测本机是否可调用 tesseract。"""
    global _tesseract_checked, _tesseract_usable
    if _tesseract_checked:
        return _tesseract_usable
    _tesseract_checked = True
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        _tesseract_usable = True
    except Exception:
        _tesseract_usable = False
    return _tesseract_usable


def _cell_esc(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).replace("\n", " ").replace("|", "\\|").strip()
    return s


def _rows_to_markdown_table(rows: list[list[Any]]) -> str:
    """将 Table.extract() 的二维列表转为 GitHub 风格 Markdown 表。"""
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    norm: list[list[str]] = []
    for r in rows:
        cells = [_cell_esc(r[i]) if i < len(r) else "" for i in range(ncols)]
        norm.append(cells)
    header = norm[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * ncols) + " |",
    ]
    for row in norm[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def extract_tables_markdown(page: Any) -> str:
    """抽取当前页表格为 Markdown；不支持或失败时返回空串。"""
    if not _tables_enabled():
        return ""
    find = getattr(page, "find_tables", None)
    if not callable(find):
        return ""
    try:
        finder = find()
    except Exception:
        return ""
    tables = getattr(finder, "tables", None)
    if not tables:
        return ""
    parts: list[str] = []
    for tab in tables:
        try:
            rows = tab.extract()
        except Exception:
            continue
        if not rows:
            continue
        md = _rows_to_markdown_table(rows)
        if md:
            parts.append(md)
    return "\n\n".join(parts).strip()


def ocr_page_raster(page: fitz.Page) -> str:
    """整页光栅化后送 Tesseract；调用方需已确认 `_tesseract_available()`。"""
    import pytesseract
    from PIL import Image

    zoom = _ocr_zoom()
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    lang = _ocr_lang()
    return pytesseract.image_to_string(img, lang=lang)


def process_pdf_page(page: fitz.Page) -> str:
    """单页完整流水线：表格 Markdown + 嵌入文本 或 OCR 文本。"""
    table_block = extract_tables_markdown(page)
    embedded = ""
    try:
        embedded = (page.get_text("text") or "").strip()
    except Exception:
        embedded = ""

    body = embedded
    min_chars = _ocr_min_chars()
    if _ocr_enabled() and len(embedded) < min_chars and _tesseract_available():
        try:
            ocr_text = ocr_page_raster(page).strip()
            # 扫描件嵌入层通常极短：有 OCR 结果则采用（即使略短于 embedded 的噪声）
            if ocr_text:
                body = ocr_text
        except Exception:
            body = embedded

    parts = [p for p in (table_block, body) if p]
    return "\n\n".join(parts).strip()
