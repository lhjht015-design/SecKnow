from __future__ import annotations

import math
import re
import shutil
import unicodedata
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol

import fitz
import numpy as np

from secknow.domain.ids import normalize_code_text, normalize_text, stable_id
from secknow.domain.models import BoundingBox, ElementKind, ParsedElement, SourceDocument

CJK = re.compile(r"[㐀-鿿]")
PRINTABLE = re.compile(r"[\w㐀-鿿]", re.UNICODE)
REPLACEMENT = re.compile(r"[�\x00-\x08\x0b\x0c\x0e-\x1f]")
SUSPICIOUS_GLYPHS = frozenset("■□●○△▽◇◆◊◉◎⊙∩∪≡≈∏∑∫√∞ⅢⅣⅤⅥⅦⅧⅨⅩ￡￠")
QUALITY_WINDOW_CHARS = 240
QUALITY_WINDOW_STEP = 120
QUALITY_WINDOW_MIN_CHARS = 96


@dataclass(frozen=True)
class TextQuality:
    score: float
    character_count: int
    replacement_ratio: float
    repetition_ratio: float
    disposition: str
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class PageExtraction:
    page_number: int
    text: str
    quality: TextQuality
    method: str
    width: float
    height: float
    diagnostics: tuple[str, ...] = ()
    segments: tuple[tuple[str, str], ...] = ()


MONO_FONT = re.compile(r"mono|courier|consol|menlo|monaco|code|fixed", re.IGNORECASE)


def _strip_code_literals(line: str, in_block_comment: bool) -> tuple[str, bool]:
    """去掉字符串与注释后再数括号，避免字面量里的 { } 干扰缩进。"""
    output: list[str] = []
    index = 0
    quote = ""
    while index < len(line):
        char = line[index]
        if in_block_comment:
            if line.startswith("*/", index):
                in_block_comment = False
                index += 2
                continue
            index += 1
            continue
        if quote:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = ""
            index += 1
            continue
        if line.startswith("//", index):
            break
        if line.startswith("/*", index):
            in_block_comment = True
            index += 2
            continue
        if char in "\"'":
            quote = char
            index += 1
            continue
        output.append(char)
        index += 1
    return "".join(output), in_block_comment


def reindent_c_like(text: str, unit: int = 4) -> str:
    """按花括号层级重建 C/C++/Java 风格代码的缩进。

    PDF 文本层通常把每一行都放在同一个 x 坐标，原始缩进没有编码；
    能可靠恢复的是“块结构”——用 { } 的深度重排。
    没有花括号的代码（汇编、Python）保持原样。
    """
    if text.count("{") < 1 or text.count("}") < 1:
        return text
    result: list[str] = []
    depth = 0
    in_block_comment = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            result.append("")
            continue
        code, in_block_comment = _strip_code_literals(stripped, in_block_comment)
        leading_close = code.lstrip().startswith(("}", ")", "]"))
        line_depth = max(0, depth - (1 if leading_close else 0))
        result.append(" " * (unit * line_depth) + stripped)
        depth = max(0, depth + code.count("{") - code.count("}"))
    return "\n".join(result)


def build_pdf_segments(page: fitz.Page) -> tuple[tuple[str, str], ...]:
    """按字体把页面拆成 code / paragraph 片段。

    等宽字体行归为 code，连续同类行合并；code 片段按花括号重建缩进。
    """
    segments: list[list[Any]] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            raw = "".join(span["text"] for span in spans).rstrip()
            if not raw.strip():
                continue
            # 一行里只要出现等宽字体就按代码处理：代码块里的中文注释
            # 通常用中文字体渲染，按“主导字体”会把注释行拆成正文。
            kind = (
                "code"
                if any(MONO_FONT.search(str(span.get("font", ""))) for span in spans)
                else "paragraph"
            )
            if segments and segments[-1][0] == kind:
                segments[-1][1].append(raw)
            else:
                segments.append([kind, [raw]])
    return tuple(
        (str(kind), reindent_c_like("\n".join(lines)) if kind == "code" else "\n".join(lines))
        for kind, lines in segments
    )


class OcrEngine(Protocol):
    def extract(self, page: fitz.Page, dpi: int) -> tuple[str, float]: ...


@dataclass(frozen=True)
class OcrCandidate:
    text: str
    confidence: float
    quality: TextQuality
    psm: int
    preprocessing: str

    @property
    def score(self) -> float:
        return self.quality.score * (0.65 + 0.35 * self.confidence)


class TesseractOcr:
    def __init__(
        self,
        languages: str = "chi_sim+eng",
        executable: Path | None = None,
        tessdata_dir: Path | None = None,
    ) -> None:
        self.languages = languages
        self.executable = executable
        self.tessdata_dir = tessdata_dir

    @property
    def command(self) -> str | None:
        if self.executable is not None:
            executable = self.executable.expanduser().resolve()
            return str(executable) if executable.is_file() else None
        return shutil.which("tesseract")

    @property
    def available(self) -> bool:
        return (
            self.command is not None
            and find_spec("pytesseract") is not None
            and find_spec("PIL") is not None
        )

    def extract(self, page: fitz.Page, dpi: int = 240) -> tuple[str, float]:
        command = self.command
        if command is None:
            raise RuntimeError("Tesseract executable is unavailable")
        try:
            import pytesseract
            from PIL import Image, ImageEnhance, ImageFilter, ImageOps
        except ImportError as error:
            raise RuntimeError("Install SecKnow with the 'ocr' extra for OCR") from error
        pytesseract.pytesseract.tesseract_cmd = command
        scale = dpi / 72
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        image = ImageOps.exif_transpose(image)
        image = ImageOps.grayscale(image)
        image = ImageOps.autocontrast(image, cutoff=1)
        image = self._correct_orientation(image, pytesseract)
        image = self._deskew(image)

        candidates: list[OcrCandidate] = []
        for psm in (3, 6):
            candidate = self._recognize(image, psm, "grayscale", pytesseract)
            candidates.append(candidate)
            if candidate.score >= 0.72:
                return candidate.text, candidate.confidence

        enhanced = ImageEnhance.Contrast(image).enhance(1.45)
        enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
        threshold = self._otsu_threshold(np.asarray(enhanced, dtype=np.uint8))
        binary = enhanced.point(lambda value: 255 if value >= threshold else 0)
        for psm in (6, 11):
            candidate = self._recognize(binary, psm, "binary", pytesseract)
            candidates.append(candidate)
            if candidate.score >= 0.72:
                return candidate.text, candidate.confidence

        best = max(candidates, key=lambda candidate: candidate.score)
        return best.text, best.confidence

    def _tessdata_config(self) -> str:
        if self.tessdata_dir is None:
            return ""
        tessdata = self.tessdata_dir.expanduser().resolve()
        if not tessdata.is_dir():
            raise RuntimeError(f"Tesseract language data is unavailable: {tessdata}")
        return f' --tessdata-dir "{tessdata}"'

    def _recognize(
        self,
        image: Any,
        psm: int,
        preprocessing: str,
        pytesseract: Any,
    ) -> OcrCandidate:
        config = f"--oem 1 --psm {psm}{self._tessdata_config()}"
        data = pytesseract.image_to_data(
            image,
            lang=self.languages,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        words: list[str] = []
        confidences: list[float] = []
        for word, confidence in zip(data["text"], data["conf"], strict=True):
            word = str(word).strip()
            try:
                score = float(confidence)
            except (TypeError, ValueError):
                continue
            if word:
                words.append(word)
                if score >= 0:
                    confidences.append(score / 100)
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        text = " ".join(words)
        return OcrCandidate(text, confidence, score_text(text), psm, preprocessing)

    def _correct_orientation(self, image: Any, pytesseract: Any) -> Any:
        try:
            output = pytesseract.image_to_osd(
                image,
                config=self._tessdata_config().strip(),
                output_type=pytesseract.Output.DICT,
            )
            rotation = int(output.get("rotate", 0)) % 360
            confidence = float(output.get("orientation_conf", 0))
        except (RuntimeError, TypeError, ValueError, pytesseract.TesseractError):
            return image
        if rotation in {90, 180, 270} and confidence >= 1:
            return image.rotate(-rotation, expand=True, fillcolor=255)
        return image

    @staticmethod
    def _deskew(image: Any) -> Any:
        from PIL import Image

        grayscale = np.asarray(image, dtype=np.uint8)
        mask = grayscale < TesseractOcr._otsu_threshold(grayscale)
        coordinates = np.column_stack(np.nonzero(mask))
        if len(coordinates) < 80:
            return image
        y_coordinates = coordinates[:, 0]
        x_coordinates = coordinates[:, 1]
        sample_step = max(1, len(coordinates) // 20000)
        covariance = np.cov(
            x_coordinates[::sample_step],
            y_coordinates[::sample_step],
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal = eigenvectors[:, int(np.argmax(eigenvalues))]
        angle = math.degrees(math.atan2(principal[1], principal[0]))
        while angle <= -45:
            angle += 90
        while angle > 45:
            angle -= 90
        if abs(angle) < 0.35 or abs(angle) > 8:
            return image
        return image.rotate(
            -angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=255,
        )

    @staticmethod
    def _otsu_threshold(grayscale: np.ndarray) -> int:
        histogram = np.bincount(grayscale.ravel(), minlength=256).astype(np.float64)
        total = histogram.sum()
        if total == 0:
            return 127
        cumulative_weight = np.cumsum(histogram)
        cumulative_mean = np.cumsum(histogram * np.arange(256))
        global_mean = cumulative_mean[-1]
        denominator = cumulative_weight * (total - cumulative_weight)
        numerator = (global_mean * cumulative_weight - cumulative_mean * total) ** 2
        variance = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        return min(255, int(np.argmax(variance)) + 1)


class RapidOcrEngine:
    """基于 RapidOCR(ONNXRuntime) 的离线 OCR 后端。

    与 TesseractOcr 实现同一个 OcrEngine 协议，可直接互换。
    选用它的原因：不需要安装任何外部可执行文件（pip 即可），
    权重随包分发、运行时完全离线，且对中文扫描件的识别质量优于 tesseract 的 chi_sim。
    本机安全策略禁止运行下载的安装包，因此 Tesseract 路线不可用。
    """

    def __init__(self, min_confidence: float = 0.40, max_side: int = 2400) -> None:
        self.min_confidence = min_confidence
        self.max_side = max_side
        self._engine: Any = None

    @property
    def available(self) -> bool:
        return find_spec("rapidocr_onnxruntime") is not None and find_spec("PIL") is not None

    def _load(self) -> Any:
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as error:
                raise RuntimeError("RapidOCR backend is unavailable") from error
            self._engine = RapidOCR()
        return self._engine

    def _render(self, page: fitz.Page, dpi: int, max_side: int) -> Any:  # noqa: ANN401
        from PIL import Image

        scale = dpi / 72
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        # 超长边会显著拖慢识别、并让 ONNX 侧申请过大的张量，等比缩到上限以内
        if max(image.size) > max_side:
            ratio = max_side / max(image.size)
            image = image.resize(
                (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
            )
        return image

    def extract(self, page: fitz.Page, dpi: int = 240) -> tuple[str, float]:
        if not self.available:
            raise RuntimeError("RapidOCR backend is unavailable")
        import numpy as np

        # 逐级降分辨率重试。ONNX Runtime 在个别超大页面上会 bad allocation，
        # 若把它抛给上层，整本几百页的书都会失败（这正是本项目踩过的坑）。
        # 降到最后一个档位仍失败时，才抛 RuntimeError 让上层把**这一页**隔离。
        last_error: Exception | None = None
        result = None
        for cap in (self.max_side, 1600, 1200):
            try:
                image = self._render(page, dpi, cap)
                result, _elapsed = self._load()(np.asarray(image))
                last_error = None
                break
            except Exception as error:  # noqa: BLE001 - ONNX 的异常类型不稳定，一律降级重试
                last_error = error
        if last_error is not None:
            raise RuntimeError(
                f"RapidOCR failed on this page after downscaling: "
                f"{type(last_error).__name__}: {last_error}"
            )
        if not result:
            return "", 0.0
        lines: list[str] = []
        scores: list[float] = []
        for item in result:
            # RapidOCR 每项为 [box, text, score]
            if len(item) < 3:
                continue
            text = str(item[1]).strip()
            score = float(item[2])
            if text and score >= self.min_confidence:
                lines.append(text)
                scores.append(score)
        if not lines:
            return "", 0.0
        return "\n".join(lines), sum(scores) / len(scores)


def select_ocr_engine(
    backend: str = "auto",
    *,
    languages: str = "chi_sim+eng",
    executable: Path | None = None,
    tessdata_dir: Path | None = None,
) -> OcrEngine | None:
    """按配置挑选可用的 OCR 后端，都不可用时返回 None。

    backend 取值：
        auto      —— 优先 RapidOCR（无需外部可执行文件），退回 Tesseract
        rapidocr  —— 只用 RapidOCR
        tesseract —— 只用 Tesseract
        none      —— 关闭 OCR
    """
    if backend == "none":
        return None
    if backend in {"auto", "rapidocr"}:
        rapid = RapidOcrEngine()
        if rapid.available:
            return rapid
        if backend == "rapidocr":
            return None
    if backend in {"auto", "tesseract"}:
        tesseract = TesseractOcr(languages, executable, tessdata_dir)
        if tesseract.available:
            return tesseract
    return None


def _script_transition_ratio(text: str) -> float:
    scripts: list[str] = []
    for character in text:
        if "㐀" <= character <= "鿿":
            scripts.append("cjk")
        elif character.isascii() and character.isalpha():
            scripts.append("latin")
        elif character.isdigit():
            scripts.append("digit")
    if len(scripts) < 2:
        return 0.0
    transitions = sum(
        previous != current
        for previous, current in zip(scripts[:-1], scripts[1:], strict=True)
    )
    return transitions / (len(scripts) - 1)


def _local_font_map_gibberish(text: str) -> tuple[float, float, float, float] | None:
    visible = "".join(character for character in text if not character.isspace())
    if len(visible) < QUALITY_WINDOW_MIN_CHARS:
        return None
    strongest: tuple[float, float, float, float] | None = None
    last_start = max(0, len(visible) - QUALITY_WINDOW_CHARS)
    starts = {*range(0, last_start + 1, QUALITY_WINDOW_STEP), last_start}
    for start in sorted(starts):
        window = visible[start : start + QUALITY_WINDOW_CHARS]
        if len(window) < QUALITY_WINDOW_MIN_CHARS:
            continue
        suspicious_ratio = sum(
            character in SUSPICIOUS_GLYPHS for character in window
        ) / len(window)
        symbol_ratio = sum(
            unicodedata.category(character) in {"Sm", "So", "Sk"}
            for character in window
        ) / len(window)
        punctuation_ratio = sum(
            unicodedata.category(character).startswith("P") for character in window
        ) / len(window)
        transition_ratio = _script_transition_ratio(window)
        detected = (
            (suspicious_ratio >= 0.025 and transition_ratio >= 0.30)
            or suspicious_ratio >= 0.12
            or (
                symbol_ratio >= 0.09
                and punctuation_ratio >= 0.20
                and transition_ratio >= 0.20
            )
        )
        signal = (
            suspicious_ratio,
            transition_ratio,
            symbol_ratio,
            punctuation_ratio,
        )
        if detected and (strongest is None or sum(signal) > sum(strongest)):
            strongest = signal
    return strongest


def score_text(text: str, page_area: float = 1.0) -> TextQuality:
    normalized = normalize_text(text)
    characters = len(normalized)
    if not normalized:
        return TextQuality(0.0, 0, 1.0, 1.0, "ocr_required", ("empty-text",))
    replacements = len(REPLACEMENT.findall(normalized))
    replacement_ratio = replacements / max(characters, 1)
    printable = len(PRINTABLE.findall(normalized)) / max(characters, 1)
    cjk = len(CJK.findall(normalized))
    unique_lines = {line.strip() for line in normalized.splitlines() if line.strip()}
    line_count = max(1, sum(1 for line in normalized.splitlines() if line.strip()))
    repetition_ratio = 1 - len(unique_lines) / line_count
    density = min(1.0, math.log1p(characters) / math.log(1501))
    language_plausibility = min(1.0, (printable * 0.8) + (min(cjk, 20) / 100))
    visible = sum(not character.isspace() for character in normalized)
    suspicious_ratio = sum(
        character in SUSPICIOUS_GLYPHS for character in normalized
    ) / max(visible, 1)
    symbol_ratio = sum(
        unicodedata.category(character) in {"Sm", "So", "Sk"}
        for character in normalized
    ) / max(visible, 1)
    script_transition_ratio = _script_transition_ratio(normalized)
    local_gibberish = _local_font_map_gibberish(normalized)
    mapped_gibberish = (
        (suspicious_ratio >= 0.025 and script_transition_ratio >= 0.24)
        or (symbol_ratio >= 0.09 and script_transition_ratio >= 0.29)
        or suspicious_ratio >= 0.10
        or local_gibberish is not None
    )
    score = (
        0.40 * density
        + 0.35 * language_plausibility
        + 0.15 * (1 - min(1.0, replacement_ratio * 20))
        + 0.10 * (1 - repetition_ratio)
    )
    if mapped_gibberish:
        score = min(score, 0.25)
    diagnostics: list[str] = []
    if characters < 40:
        diagnostics.append("low-character-count")
    if replacement_ratio > 0.01:
        diagnostics.append("replacement-characters")
    if printable < 0.55:
        diagnostics.append("low-printable-ratio")
    if repetition_ratio > 0.6:
        diagnostics.append("repeated-lines")
    if mapped_gibberish:
        diagnostics.append(
            "suspected-font-map-gibberish:"
            f"symbols={symbol_ratio:.3f},"
            f"mapped={suspicious_ratio:.3f},"
            f"transitions={script_transition_ratio:.3f}"
        )
    if local_gibberish is not None:
        (
            local_suspicious,
            local_transitions,
            local_symbols,
            local_punctuation,
        ) = local_gibberish
        diagnostics.append(
            "localized-font-map-gibberish:"
            f"mapped={local_suspicious:.3f},"
            f"symbols={local_symbols:.3f},"
            f"punctuation={local_punctuation:.3f},"
            f"transitions={local_transitions:.3f}"
        )
    disposition = "native" if score >= 0.52 and characters >= 40 else "ocr_required"
    return TextQuality(
        round(max(0.0, min(score, 1.0)), 6),
        characters,
        replacement_ratio,
        repetition_ratio,
        disposition,
        tuple(diagnostics),
    )


class PdfExtractor:
    def __init__(self, ocr: OcrEngine | None = None, dpi: int = 240) -> None:
        self.ocr = ocr
        self.dpi = dpi

    def extract(self, document: SourceDocument) -> tuple[PageExtraction, ...]:
        pages: list[PageExtraction] = []
        with fitz.open(document.absolute_path) as pdf:
            for index, page in enumerate(pdf):
                native_text = page.get_text("text", sort=True)
                native_quality = score_text(native_text, page.rect.width * page.rect.height)
                selected_text = native_text
                quality = native_quality
                method = "pymupdf"
                diagnostics = list(native_quality.diagnostics)
                if native_quality.disposition == "ocr_required":
                    if self.ocr is None:
                        diagnostics.append("ocr-not-configured")
                    else:
                        try:
                            ocr_text, ocr_confidence = self.ocr.extract(page, self.dpi)
                            ocr_quality = score_text(ocr_text, page.rect.width * page.rect.height)
                            adjusted_ocr = ocr_quality.score * (0.65 + 0.35 * ocr_confidence)
                            if adjusted_ocr > native_quality.score:
                                selected_text = ocr_text
                                quality = TextQuality(
                                    adjusted_ocr,
                                    ocr_quality.character_count,
                                    ocr_quality.replacement_ratio,
                                    ocr_quality.repetition_ratio,
                                    "ocr" if adjusted_ocr >= 0.45 else "quarantine",
                                    (
                                        *ocr_quality.diagnostics,
                                        f"ocr-confidence:{ocr_confidence:.3f}",
                                    ),
                                )
                                method = "tesseract"
                            else:
                                diagnostics.append("ocr-not-better")
                        except RuntimeError as error:
                            diagnostics.append(f"ocr-unavailable:{error}")
                if quality.score < 0.3:
                    quality = TextQuality(
                        quality.score,
                        quality.character_count,
                        quality.replacement_ratio,
                        quality.repetition_ratio,
                        "quarantine",
                        quality.diagnostics,
                    )
                pages.append(
                    PageExtraction(
                        page_number=index + 1,
                        text=normalize_text(selected_text),
                        quality=quality,
                        method=method,
                        width=page.rect.width,
                        height=page.rect.height,
                        diagnostics=tuple(diagnostics),
                        segments=(build_pdf_segments(page) if method == "pymupdf" else ()),
                    )
                )
        return tuple(pages)

    def to_elements(
        self,
        document: SourceDocument,
        pages: tuple[PageExtraction, ...],
    ) -> tuple[ParsedElement, ...]:
        elements: list[ParsedElement] = []
        order = 0
        for page in pages:
            box = BoundingBox(
                x0=0,
                y0=0,
                x1=page.width,
                y1=page.height,
                page_width=page.width,
                page_height=page.height,
            )
            segments = page.segments or (("paragraph", page.text),)
            for segment_index, (kind, text) in enumerate(segments):
                normalized = (
                    normalize_code_text(text)
                    if kind == "code"
                    else normalize_text(text)
                )
                if not normalized:
                    continue
                elements.append(
                    ParsedElement(
                        element_id=stable_id(
                            "element",
                            document.source_id,
                            document.version_id,
                            page.page_number,
                            segment_index,
                            normalized[:128],
                        ),
                        source_id=document.source_id,
                        source_version=document.version_id,
                        kind=ElementKind.CODE if kind == "code" else ElementKind.PARAGRAPH,
                        text=normalized,
                        order=order,
                        page=page.page_number,
                        bounding_box=box,
                        parser=page.method,
                        extraction_confidence=page.quality.score,
                        diagnostics=tuple(
                            dict.fromkeys((*page.quality.diagnostics, *page.diagnostics))
                        ),
                        metadata={"disposition": page.quality.disposition},
                    )
                )
                order += 1
        return tuple(elements)


def parse_with_unstructured(document: SourceDocument) -> tuple[ParsedElement, ...]:
    try:
        from unstructured.partition.auto import partition
    except ImportError as error:
        raise RuntimeError("Install SecKnow with the 'parse' extra for Unstructured") from error
    elements: list[ParsedElement] = []
    for order, element in enumerate(partition(filename=document.absolute_path)):
        metadata = element.metadata.to_dict()
        category = str(getattr(element, "category", "unknown")).casefold()
        kind = {
            "title": ElementKind.TITLE,
            "header": ElementKind.HEADING,
            "narrativetext": ElementKind.PARAGRAPH,
            "listitem": ElementKind.LIST_ITEM,
            "table": ElementKind.TABLE,
        }.get(category, ElementKind.UNKNOWN)
        text = normalize_text(str(element))
        if not text:
            continue
        page = metadata.get("page_number")
        elements.append(
            ParsedElement(
                element_id=stable_id(
                    "element", document.source_id, document.version_id, order, text[:128]
                ),
                source_id=document.source_id,
                source_version=document.version_id,
                kind=kind,
                text=text,
                order=order,
                page=page if isinstance(page, int) else None,
                parser="unstructured",
                extraction_confidence=1.0,
                metadata=metadata,
            )
        )
    return tuple(elements)
