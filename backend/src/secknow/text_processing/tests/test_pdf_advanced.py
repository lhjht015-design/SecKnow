from __future__ import annotations

"""pdf_advanced 纯逻辑单测（不打开真实 PDF、不依赖 Tesseract）。"""

from secknow.text_processing.loaders.pdf_advanced import _rows_to_markdown_table


def test_rows_to_markdown_table() -> None:
    rows = [["A", "B"], ["1", "2"]]
    md = _rows_to_markdown_table(rows)
    assert "| A | B |" in md
    assert "| --- | --- |" in md
    assert "| 1 | 2 |" in md
