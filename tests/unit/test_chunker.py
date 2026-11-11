from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import fitz
import numpy as np

from secknow.domain.ids import sha256_bytes
from secknow.domain.models import DocumentFormat, ElementKind, ParsedElement, SourceDocument
from secknow.indexing.chunker import SemanticChunker
from secknow.ingestion.document_parser import (
    PageExtraction,
    PdfExtractor,
    TesseractOcr,
    TextQuality,
    score_text,
)


def document() -> SourceDocument:
    return SourceDocument(
        source_id="document:test",
        version_id="version:test",
        title="测试文档",
        relative_path="test.md",
        absolute_path="C:/test.md",
        format=DocumentFormat.MARKDOWN,
        media_type="text/markdown",
        size_bytes=10,
        sha256=sha256_bytes(b"test"),
        modified_at=datetime.now(UTC),
    )


def test_chunker_preserves_citation_and_links() -> None:
    elements = tuple(
        ParsedElement(
            element_id=f"element:{index}",
            source_id="document:test",
            source_version="version:test",
            kind=ElementKind.PARAGRAPH,
            text=("栈溢出会覆盖返回地址。ROP 可以复用代码片段。" * 8),
            order=index,
            page=index + 1,
            section_path=("Pwn",),
            parser="test",
        )
        for index in range(2)
    )
    chunks = SemanticChunker(chunk_size=40, chunk_overlap=4).chunk(document(), elements)
    assert len(chunks) >= 2
    assert chunks[0].next_chunk_id == chunks[1].chunk_id
    assert chunks[1].previous_chunk_id == chunks[0].chunk_id
    assert chunks[0].citation.page_start == 1
    assert all(chunk.section_path == ("Pwn",) for chunk in chunks)


def test_pdf_quality_rejects_font_map_gibberish_but_keeps_normal_text() -> None:
    corrupted = (
        "第7章漏洞利用开发|5| ……耻延…璃溶===页Pˉ±亚■…pp↓邑…… "
        "bre己k s十白了七 c○tjnue 卜NOwseth工e己kp○土n亡○nsh己reα] "
        "土b工己ryr○七土nes b工e己km己11○C Dre己Kfree c○n七jnue P■■器"
    )
    normal = (
        "第7章漏洞利用开发。pwntools 的 DynELF 模块可以解析 libc 符号，"
        "ROP 链通过 pop rdi; ret 设置参数，再调用 system 函数。"
    )

    corrupted_quality = score_text(corrupted)
    normal_quality = score_text(normal)

    assert corrupted_quality.disposition == "ocr_required"
    assert any(
        diagnostic.startswith("suspected-font-map-gibberish:")
        for diagnostic in corrupted_quality.diagnostics
    )
    assert normal_quality.disposition == "native"


def test_pdf_quality_rejects_localized_font_map_gibberish() -> None:
    normal_prefix = (
        "格式化字符串漏洞允许攻击者读取栈中数据，理解调用约定后可以确认参数位置。"
        "程序会依次读取格式字符串，并根据转换说明输出对应内容。"
    ) * 5
    localized_corruption = (
        "gef＞de工efe工enCeSeSP 肖蔼0苹f苹cdα0｜仆0X00O0 "
        "fcdd4｜卢→倘He11。W。r1d!! ■○∩≡ p七w工n"
    ) * 4
    quality = score_text(normal_prefix + localized_corruption)

    assert quality.disposition == "ocr_required"
    assert any(
        diagnostic.startswith("localized-font-map-gibberish:")
        for diagnostic in quality.diagnostics
    )


def test_pdf_quality_rejects_symbol_and_punctuation_heavy_local_corruption() -> None:
    normal_prefix = (
        "该模块用于操作 ELF 文件，包括符号查找、虚拟内存、文件偏移以及保存二进制文件。"
        "程序可以读取指定地址的数据，并按章节说明继续调试。"
    ) * 4
    localized_corruption = (
        "pW∩||b。e|f′;>>>.P工A七B己C土D1丁E2丅F3"
        "·write（address，data）：P工A七B己C土D1丁E2丅F3；≫≫≫……＞＞∩≡"
        "｛［（＞＞＞｜｜＝＝）］｝·section（name）：P工A七B己C土D1丁E2丅F3；"
    ) * 5

    quality = score_text(normal_prefix + localized_corruption)

    assert quality.disposition == "ocr_required"
    assert any(
        "symbols=" in diagnostic and "punctuation=" in diagnostic
        for diagnostic in quality.diagnostics
        if diagnostic.startswith("localized-font-map-gibberish:")
    )


def test_pdf_quality_keeps_long_normal_mixed_security_text() -> None:
    normal = (
        "在 x86-64 架构中，ROP 链通过 pop rdi; ret 设置第一个参数，"
        "随后调用 system('/bin/sh')。寄存器包括 RAX、RSP、RIP，"
        "地址示例为 0x7ffff7dd0000，GDB 命令是 x/20gx $rsp。"
    ) * 8

    assert score_text(normal).disposition == "native"


def test_pdf_element_diagnostics_are_unique() -> None:
    quality = TextQuality(
        score=0.25,
        character_count=12,
        replacement_ratio=0,
        repetition_ratio=0,
        disposition="quarantine",
        diagnostics=("font-map-corruption",),
    )
    page = PageExtraction(
        page_number=1,
        text="damaged text",
        quality=quality,
        method="pymupdf",
        width=100,
        height=100,
        diagnostics=("font-map-corruption", "ocr-not-configured"),
    )

    element = PdfExtractor().to_elements(document(), (page,))[0]

    assert element.diagnostics == ("font-map-corruption", "ocr-not-configured")


def test_pdf_ocr_runs_only_for_bad_pages_and_keeps_one_result(tmp_path: Path) -> None:
    pdf_path = tmp_path / "mixed.pdf"
    pdf = fitz.open()
    good_page = pdf.new_page()
    good_page.insert_text(
        (72, 72),
        "ROP chains reuse existing instructions and preserve calling conventions. " * 4,
    )
    pdf.new_page()
    pdf.save(pdf_path)
    pdf.close()
    source = document().model_copy(
        update={
            "absolute_path": str(pdf_path),
            "relative_path": "mixed.pdf",
            "format": DocumentFormat.PDF,
            "media_type": "application/pdf",
        }
    )

    class RecordingOcr:
        def __init__(self) -> None:
            self.pages: list[int] = []

        def extract(self, page: fitz.Page, dpi: int) -> tuple[str, float]:
            self.pages.append(page.number + 1)
            return "Recovered OCR text explains stack frames and return addresses. " * 4, 0.96

    ocr = RecordingOcr()
    pages = PdfExtractor(ocr).extract(source)

    assert ocr.pages == [2]
    assert pages[0].method == "pymupdf"
    assert pages[1].method == "tesseract"
    assert pages[1].text.startswith("Recovered OCR text")
    assert pages[1].quality.disposition == "ocr"


def test_tesseract_otsu_threshold_preserves_black_and_white_pixels() -> None:
    grayscale = np.array([[0, 0, 255, 255]], dtype=np.uint8)

    threshold = TesseractOcr._otsu_threshold(grayscale)
    binary = grayscale >= threshold

    assert 0 < threshold <= 255
    assert binary.tolist() == [[False, False, True, True]]


def test_tesseract_explicit_executable_must_exist(tmp_path: Path) -> None:
    executable = tmp_path / "tesseract.exe"
    executable.write_bytes(b"placeholder")

    assert TesseractOcr(executable=executable).command == str(executable.resolve())
    assert TesseractOcr(executable=tmp_path / "missing.exe").command is None


def test_tesseract_orientation_uses_configured_tessdata(tmp_path: Path) -> None:
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()

    class FakePytesseract:
        class Output:
            DICT = "dict"

        TesseractError = RuntimeError

        def __init__(self) -> None:
            self.config = ""

        def image_to_osd(
            self,
            image: object,
            *,
            config: str,
            output_type: str,
        ) -> dict[str, object]:
            self.config = config
            return {"rotate": 0, "orientation_conf": 10}

    image = object()
    pytesseract = FakePytesseract()
    corrected = TesseractOcr(tessdata_dir=tessdata)._correct_orientation(image, pytesseract)

    assert corrected is image
    assert pytesseract.config == f'--tessdata-dir "{tessdata.resolve()}"'


def test_chunker_excludes_pages_requiring_ocr() -> None:
    quarantined = ParsedElement(
        element_id="element:bad",
        source_id="document:test",
        source_version="version:test",
        kind=ElementKind.PARAGRAPH,
        text="mapped glyph noise ■○∩≡ p七w工n",
        order=0,
        page=1,
        parser="pymupdf",
        metadata={"disposition": "ocr_required"},
    )
    native = quarantined.model_copy(
        update={
            "element_id": "element:good",
            "text": "ROP uses existing instruction sequences.",
            "order": 1,
            "page": 2,
            "metadata": {"disposition": "native"},
        }
    )

    chunks = SemanticChunker(chunk_size=128, chunk_overlap=16).chunk(
        document(),
        (quarantined, native),
    )

    assert len(chunks) == 1
    assert chunks[0].text == native.text
    assert chunks[0].citation.page_start == 2
