from __future__ import annotations

"""按业务文件类型串联清洗（流水线与 `load_document(normalize=True)` 的唯一入口）。

**推荐阅读顺序**：`basic.py` → `normalize.py` → `pdf.py`（仅 PDF）→ `markdown.py` / `code.py`。

**当前流水线顺序（重要）**
1. `basic_clean`：换行、控制字符、空行、BOM——**与文件类型无关**。
2. `clean_pdf_document`：仅当 `file_type == "pdf"`。依赖 `load_pdf` 在页间插入
   `PDF_PAGE_BOUNDARY`，用于页眉页脚启发式；无界标则原样跳过。
3. `normalize_for_file_type`：标点与 Unicode 空白；**代码**走保守分支（不映射 FF01–FF5E）。
4. 类型收尾：`clean_markdown` / `clean_code`；`text`/`docx` 等无额外步骤。

**改动历史摘要**
- 相对「仅 basic_clean」版本：补全 PDF 页界、全角标点/宽空格、MD 注释与代码/MD 专用轻量规则。
"""

from .basic import basic_clean
from .code import clean_code
from .markdown import clean_markdown
from .normalize import normalize_for_file_type
from .pdf import clean_pdf_document


def clean_document_text(text: str, file_type: str) -> str:
    """对原始抽取文本做完整清洗链，返回可供分块的正文。"""
    base = basic_clean(text)
    if not base:
        return base

    if file_type == "pdf":
        base = clean_pdf_document(base)

    base = normalize_for_file_type(base, file_type=file_type)

    if file_type == "markdown":
        return clean_markdown(base)
    if file_type == "code":
        return clean_code(base)
    return base
