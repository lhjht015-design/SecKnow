from __future__ import annotations

"""清洗层单测：覆盖 basic / normalize / markdown / pdf 页界与页眉启发式。"""

from secknow.text_processing.cleaners.basic import basic_clean
from secknow.text_processing.cleaners.document_clean import clean_document_text
from secknow.text_processing.cleaners.markdown import clean_markdown
from secknow.text_processing.cleaners.normalize import (
    map_fullwidth_ascii_block,
    normalize_for_file_type,
)
from secknow.text_processing.cleaners.pdf import PDF_PAGE_BOUNDARY, clean_pdf_document


def test_basic_strips_bom_and_control_chars() -> None:
    raw = "\ufeffline1\r\nline2\x00\x7f"
    out = basic_clean(raw)
    assert "\ufeff" not in out
    assert "\x00" not in out
    assert "line1" in out and "line2" in out


def test_map_fullwidth_ascii_block() -> None:
    # U+FF01 FULLWIDTH EXCLAMATION -> !
    assert map_fullwidth_ascii_block("\uff01") == "!"
    # 中文句号 U+3002 不在 FF01-FF5E，应保留
    assert "\u3002" in map_fullwidth_ascii_block("句\u3002")


def test_normalize_prose_vs_code() -> None:
    mixed = "A\uff01\u3000B"  # 全角叹号 + 全角空格
    prose = normalize_for_file_type(mixed, file_type="text")
    assert "! " in prose or prose.startswith("A!")
    code = normalize_for_file_type(mixed, file_type="code")
    assert "\uff01" in code
    assert "\u3000" not in code


def test_clean_markdown_strips_html_comment() -> None:
    md = "前<!-- multi\nline -->后"
    assert "<!--" not in clean_markdown(md)
    assert "前" in clean_markdown(md) and "后" in clean_markdown(md)


def test_clean_pdf_repeated_header() -> None:
    # 三页非空，首行均为相同「页眉」
    header = "CONFIDENTIAL HEADER"
    p1 = f"{header}\n正文一"
    p2 = f"{header}\n正文二"
    p3 = f"{header}\n正文三"
    raw = PDF_PAGE_BOUNDARY.join([p1, p2, p3])
    after_basic = basic_clean(raw)
    cleaned = clean_pdf_document(after_basic, repeat_ratio=0.6)
    assert header not in cleaned
    assert "正文一" in cleaned and "正文三" in cleaned


def test_clean_document_text_pdf_chain() -> None:
    # 首行需 ≥ PDF_HEADER_MIN_LINE_LEN（默认 3）才会参与页眉统计
    h = "重复页眉行"
    raw = PDF_PAGE_BOUNDARY.join([f"{h}\nX", f"{h}\nY"])
    out = clean_document_text(raw, file_type="pdf")
    assert h not in out
    assert "X" in out and "Y" in out


def test_clean_document_text_markdown_chain() -> None:
    text = "# T\n\n\n\n正文\uff01"
    out = clean_document_text(text, file_type="markdown")
    assert "<!--" not in out
    assert "!" in out or "\uff01" not in out
