from __future__ import annotations

"""PDF 加载入口（PyMuPDF + 表格 + 可选 OCR）。

**不要再在此文件堆业务逻辑**：单页处理全部在 `pdf_advanced.py`，便于单测与文档阅读。

**流程摘要**
1. `fitz.open` 打开文档；
2. 对每一页调用 `pdf_advanced.process_pdf_page`（表格 → 嵌入文 → 不足则 OCR）；
3. 用 `PDF_PAGE_BOUNDARY` 连接各页字符串，供 `cleaners/pdf.clean_pdf_document` 识别页界。

**与旧版差异**
- 曾仅用 `pdfplumber` 拼文本；现以 **PyMuPDF** 为主，支持光栅化与 `find_tables`，
  扫描件在安装了 Tesseract 时可出字。
"""

from pathlib import Path

from ..cleaners.pdf import PDF_PAGE_BOUNDARY
from ..exceptions import DocumentLoadError
from .pdf_advanced import process_pdf_page

try:
    import fitz
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "加载 PDF 需要安装 pymupdf：pip install pymupdf"
    ) from exc


def load_pdf(file_path: str | Path) -> str:
    """提取 PDF：每页结构化文本 + 可选 OCR，页间插入 `PDF_PAGE_BOUNDARY`。"""
    path = Path(file_path)
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise DocumentLoadError(f"读取 PDF 失败: {path}") from exc

    try:
        page_texts: list[str] = []
        for i in range(len(doc)):
            page = doc[i]
            try:
                page_texts.append(process_pdf_page(page))
            except Exception as exc:
                raise DocumentLoadError(
                    f"处理 PDF 第 {i + 1} 页失败: {path}: {exc}"
                ) from exc
    finally:
        doc.close()

    return PDF_PAGE_BOUNDARY.join(page_texts)
