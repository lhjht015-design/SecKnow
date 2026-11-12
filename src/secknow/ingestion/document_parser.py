from __future__ import annotations

import json
import math
import os
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
# 不解码页面里的图片：扫描书每页都是整页位图，默认 flags 会把整本书的图都解一遍。
DICT_FLAGS = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES
TABLE_CAPTION = re.compile(r"表\s*\d+[-.．]\d+")


def _table_column_bounds(
    image_path: Path,
    origin_x: float = 0.0,
    scale: float = 1.0,
) -> list[float] | None:
    """从表格裁剪图里检测竖线，返回 PDF/图像坐标下的列边界。

    有竖线的表格（绝大多数扫描书表格）用线界分列最准；
    检测不到 3 条以上竖线时返回 None，让调用方回退到词框聚类。
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return None
    try:
        with Image.open(image_path) as image:
            gray = np.asarray(image.convert("L"))
    except (OSError, ValueError):
        return None
    if gray.ndim != 2 or gray.shape[0] < 40 or gray.shape[1] < 60:
        return None
    dark = gray < 160
    height = dark.shape[0]

    # 竖线必须“连续贯穿表体”：文字笔画只有几十像素，长行程能把它们排除掉。
    # 不用横线先裁 body（有的表横线稀疏，反而会裁出窄条导致误判）。
    longest = np.zeros(dark.shape[1], dtype=np.int32)
    run = np.zeros(dark.shape[1], dtype=np.int32)
    for row in dark:
        run = (run + 1) * row
        np.maximum(longest, run, out=longest)
    mask = longest >= max(30, int(height * 0.45))
    groups: list[tuple[int, int]] = []
    start = None
    for x, flag in enumerate(mask):
        if flag and start is None:
            start = x
        elif not flag and start is not None:
            groups.append((start, x - 1))
            start = None
    if start is not None:
        groups.append((start, len(mask) - 1))
    centers = [
        (a + b) / 2 for a, b in groups
        if 0 <= b - a <= 10  # 只保留细线，排除整块深色区域
    ]
    if len(centers) < 2:
        return None
    bounds = [origin_x + center / scale for center in centers]
    if bounds[0] - origin_x > 8:
        bounds.insert(0, origin_x)
    if origin_x + gray.shape[1] / scale - bounds[-1] > 8:
        bounds.append(origin_x + gray.shape[1] / scale)
    return bounds if len(bounds) >= 3 else None


def _column_bounds_from_items(
    items: list[tuple[tuple[float, float, float, float], str, float]],
) -> list[float] | None:
    """无竖线表格：找“竖直空白走廊”作为列边界。

    统计每个 x 上被词框覆盖的行数；覆盖为 0（或只有极少数合并行跨过）
    且宽度足够的连续区间就是列间空隙。
    """
    if len(items) < 6:
        return None
    left = min(box[0] for box, _, _ in items)
    right = max(box[2] for box, _, _ in items)
    if right - left < 60:
        return None
    heights = sorted(box[3] - box[1] for box, _, _ in items)
    median_height = heights[len(heights) // 2] or 10.0
    min_gap = max(4.0, median_height * 0.35)

    size = int(right - left) + 2
    diff = [0] * (size + 1)
    for box, _, _ in items:
        start = max(0, int(box[0] - left))
        finish = min(size - 1, int(box[2] - left))
        if finish < start:
            continue
        diff[start] += 1
        diff[finish + 1] -= 1
    coverage = [0] * size
    running = 0
    for index in range(size):
        running += diff[index]
        coverage[index] = running

    # 允许极少数合并单元格/长描述跨过空隙，不把它们当成列
    tolerance = max(1, int(len(items) * 0.05))
    gaps: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(coverage):
        low = value <= tolerance
        if low and start is None:
            start = index
        elif not low and start is not None:
            if index - start >= min_gap:
                gaps.append((start, index - 1))
            start = None
    if start is not None and size - start >= min_gap:
        gaps.append((start, size - 1))
    if not gaps:
        return None
    if len(gaps) > 12:
        gaps = sorted(gaps, key=lambda gap: gap[1] - gap[0], reverse=True)[:12]
        gaps.sort()
    bounds = [left]
    bounds.extend(left + (a + b) / 2 for a, b in gaps)
    bounds.append(right)
    return bounds if len(bounds) >= 3 else None


def _items_to_markdown_table(
    items: list[tuple[tuple[float, float, float, float], str, float]],
    column_bounds: list[float] | None = None,
) -> str:
    """把带坐标的 OCR 结果按行/列重建成 Markdown 表格。"""
    if len(items) < 2:
        return "\n".join(text for _, text, _ in items)

    heights = sorted(box[3] - box[1] for box, _, _ in items)
    median_height = heights[len(heights) // 2] or 1.0

    ordered = sorted(items, key=lambda item: ((item[0][1] + item[0][3]) / 2, item[0][0]))
    rows: list[tuple[float, list[tuple[tuple[float, float, float, float], str, float]]]] = []
    for item in ordered:
        center = (item[0][1] + item[0][3]) / 2
        if not rows or center - rows[-1][0] > max(6.0, median_height * 0.6):
            rows.append((center, [item]))
        else:
            rows[-1][1].append(item)

    def row_items_sorted(row_items):
        return sorted(row_items, key=lambda item: item[0][0])

    if column_bounds is None:
        column_bounds = _column_bounds_from_items(items)
    header_items = row_items_sorted(rows[0][1]) if rows else []
    if column_bounds and len(column_bounds) >= 3:
        # 有竖线时直接按线界分列，这是最可靠的列切分
        columns = [
            [column_bounds[index], column_bounds[index + 1]]
            for index in range(len(column_bounds) - 1)
        ]
    else:
        intervals = sorted((box[0], box[2]) for box, _, _ in items)
        columns = []
        for x0, x1 in intervals:
            if columns and x0 <= columns[-1][1]:
                columns[-1][1] = max(columns[-1][1], x1)
            else:
                columns.append([x0, x1])
        if len(columns) < 2 or len(columns) < len(header_items):
            columns = []

    if columns:
        grid: list[list[str]] = []
        for _, row in rows:
            cells = [""] * len(columns)
            for box, text, _ in row_items_sorted(row):
                best_index = 0
                best_overlap = -1.0
                for index, (left, right) in enumerate(columns):
                    overlap = min(box[2], right) - max(box[0], left)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_index = index
                cells[best_index] = f"{cells[best_index]} {text}".strip()
            if any(cells):
                grid.append([cell.replace("|", "／").replace("\n", " ")
                             for cell in cells])
        column_count = len(columns)
    elif len(header_items) >= 3 and len(rows) >= 2:
        # 区间法把多列表压成了 2 列（单元格贴边）时，用表头中心重新分列
        centers = [(box[0] + box[2]) / 2 for box, _, _ in header_items]
        bounds = [
            (centers[index] + centers[index + 1]) / 2
            for index in range(len(centers) - 1)
        ]
        grid = []
        for _, row in rows:
            cells = [""] * len(centers)
            for box, text, _ in row_items_sorted(row):
                center = (box[0] + box[2]) / 2
                index = 0
                while index < len(bounds) and center > bounds[index]:
                    index += 1
                cells[index] = f"{cells[index]} {text}".strip()
            if any(cells):
                grid.append([cell.replace("|", "／").replace("\n", " ")
                             for cell in cells])
        column_count = len(centers)
    else:
        return "\n".join(
            " | ".join(text for _, text, _ in row_items_sorted(row[1]))
            for row in rows
        )
    if len(grid) >= 2 and column_count > 2:
        # 页边页码等孤立项会形成几乎全空的列，按填充率剪掉
        threshold = max(1, int(math.ceil(len(grid) * 0.2)))
        keep = [
            index for index in range(column_count)
            if sum(1 for row in grid if row[index].strip()) >= threshold
        ]
        if len(keep) >= 2:
            grid = [[row[index] for index in keep] for row in grid]
    if len(grid) < 2:
        return "\n".join(" | ".join(cell for cell in row if cell) for row in grid)

    header = grid[0]
    body = grid[1:]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _ocr_table_markdown(image_path: Path, ocr: OcrEngine | None) -> str:
    """对裁剪出来的表格图做 OCR，带缓存。"""
    if ocr is None or not image_path.exists():
        return ""
    cache = image_path.with_suffix(".ocr-v2.json")
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            return str(data.get("table") or "")
        except (OSError, ValueError):
            pass
    items: list[tuple[tuple[float, float, float, float], str, float]] = []
    try:
        with fitz.open(image_path) as image_doc:
            page = image_doc[0]
            if hasattr(ocr, "extract_items"):
                items = ocr.extract_items(page, dpi=72)  # type: ignore[attr-defined]
            else:
                text, _confidence = ocr.extract(page, dpi=72)
                items = [
                    ((0.0, float(index * 10), 1000.0, float(index * 10 + 9)), line, 1.0)
                    for index, line in enumerate(text.splitlines()) if line.strip()
                ]
    except (RuntimeError, OSError, ValueError):
        items = []
    bounds = _table_column_bounds(image_path, 0.0, 1.0)
    table = _items_to_markdown_table(items, bounds)
    if table:
        try:
            cache.write_text(json.dumps({"table": table}, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    return table


def _scan_page_table(
    document: SourceDocument,
    page: fitz.Page,
    page_number: int,
    media_dir: Path,
    ocr: OcrEngine | None,
    stored_text: str,
) -> tuple[tuple[str, str], ...]:
    """扫描书没有文本层：用 OCR 词框找题注，再按“下一段宽行”截出表格。"""
    if ocr is None or not TABLE_CAPTION.search(stored_text):
        return ()
    source_key = document.source_id.split(":")[-1]
    cache = media_dir / "scan-ocr" / source_key / f"p{page_number:04d}.items.json"
    items: list[tuple[tuple[float, float, float, float], str, float]] = []
    if cache.exists():
        try:
            raw = json.loads(cache.read_text(encoding="utf-8"))
            items = [
                ((float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                 str(text), float(score))
                for box, text, score in raw
            ]
        except (OSError, ValueError, TypeError):
            items = []
    if not items:
        if not hasattr(ocr, "extract_items"):
            return ()
        try:
            items = ocr.extract_items(page, dpi=200)  # type: ignore[attr-defined]
        except (RuntimeError, OSError, ValueError):
            return ()
        if not items:
            return ()
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(
                json.dumps([[list(box), text, score] for box, text, score in items],
                           ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    scale = 200 / 72
    pdf_items = [
        ((box[0] / scale, box[1] / scale, box[2] / scale, box[3] / scale), text, score)
        for box, text, score in items
    ]
    captions = sorted(
        (item for item in pdf_items if TABLE_CAPTION.match(item[1].strip())),
        key=lambda item: item[0][1],
    )
    if not captions:
        return ()

    ordered = sorted(pdf_items, key=lambda item: (item[0][1], item[0][0]))
    max_width = max(box[2] - box[0] for box, _, _ in pdf_items) or page.rect.width
    heights = sorted(box[3] - box[1] for box, _, _ in pdf_items)
    median_height = heights[len(heights) // 2] or 10.0
    target_dir = media_dir / "pdf-tables-scan" / source_key
    target_dir.mkdir(parents=True, exist_ok=True)
    doc_dir = media_dir.parent / Path(document.relative_path).parent

    output: list[tuple[str, str]] = []
    covered_until = 0.0
    table_index = 0
    for caption in captions:
        caption_top = caption[0][1]
        caption_bottom = caption[0][3]
        if caption_top < covered_until - 2:
            continue
        table_items: list[tuple[tuple[float, float, float, float], str, float]] = []
        end_y = page.rect.height - 24
        previous_bottom = caption_bottom
        for item in ordered:
            box, text, _ = item
            if box[1] < caption_bottom - 2:
                continue
            width = box[2] - box[0]
            gap = box[1] - previous_bottom
            looks_like_paragraph = (
                len(text) >= 12 and width >= 0.6 * max_width
            ) or (
                gap >= 0.9 * median_height and width >= 0.5 * max_width
            ) or (
                text.rstrip().endswith(("。", "："))
                and gap >= 0.5 * median_height
                and width >= 0.35 * max_width
            )
            if looks_like_paragraph:
                end_y = box[1] - 2
                break
            table_items.append(item)
            previous_bottom = max(previous_bottom, box[3])
        if len(table_items) < 3:
            continue

        x0 = min(box[0] for box, _, _ in table_items) - 4
        x1 = max(box[2] for box, _, _ in table_items) + 4
        y0 = max(0.0, caption_top - 2)
        y1 = min(page.rect.height, end_y + 2)
        rect = fitz.Rect(max(0.0, x0), y0, min(page.rect.width, x1), y1)
        if rect.width < 40 or rect.height < 20:
            continue

        table_index += 1
        target = target_dir / f"p{page_number:04d}-{table_index:02d}.png"
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, alpha=False)
        target.write_bytes(pixmap.tobytes("png"))
        relative = os.path.relpath(target, doc_dir).replace("\\", "/")
        alt = caption[1].replace("]", "）").strip()
        output.append(("image", f"![{alt}]({relative})"))
        bounds = _table_column_bounds(target, rect.x0, 2.0)
        table = _items_to_markdown_table(table_items, bounds)
        if table:
            output.append(("table", table))
        covered_until = y1
    return tuple(output)


def _image_markdown_for(
    document: SourceDocument,
    page: fitz.Page,
    page_number: int,
    media_dir: Path,
    blocks: list[dict[str, Any]] | None = None,
    ocr: OcrEngine | None = None,
) -> tuple[tuple[str, str], ...]:
    """把带“表X-Y”题注的图片裁出来，输出 Markdown 图片行。

    这些书的表格基本是位图；文本层里没有内容，只能按题注定位后裁剪。
    """
    if blocks is None:
        blocks = page.get_text("dict", flags=DICT_FLAGS).get("blocks", [])
    captions: list[tuple[str, tuple[float, float, float, float]]] = []
    for block in blocks:
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            raw = "".join(span["text"] for span in spans).strip()
            if not raw or not TABLE_CAPTION.search(raw):
                continue
            x0 = min(span["bbox"][0] for span in spans)
            y0 = min(span["bbox"][1] for span in spans)
            x1 = max(span["bbox"][2] for span in spans)
            y1 = max(span["bbox"][3] for span in spans)
            captions.append((raw, (x0, y0, x1, y1)))
    if not captions:
        return ()
    # 优先用“表X-Y 标题”这种题注行；只有“如表X-Y所示”时再用正文行
    titled = [item for item in captions if TABLE_CAPTION.match(item[0])]
    if titled:
        captions = titled

    page_area = page.rect.width * page.rect.height
    candidates: list[tuple[float, fitz.Rect]] = []
    for image in page.get_images(full=True)[:300]:
        try:
            rects = page.get_image_rects(image[0])
        except (RuntimeError, ValueError):
            continue
        for rect in rects:
            area = rect.width * rect.height
            if area < 3000 or area > page_area * 0.8:
                continue
            candidates.append((area, rect))
    if not candidates:
        return ()

    output: list[tuple[str, str]] = []
    used: set[tuple[float, float, float, float]] = set()
    target_dir = media_dir / "pdf-tables" / document.source_id.split(":")[-1]
    for caption, box in captions:
        center_y = (box[1] + box[3]) / 2
        ranked = sorted(
            candidates,
            key=lambda item: abs((item[1].y0 + item[1].y1) / 2 - center_y),
        )
        if not ranked:
            continue
        _, rect = ranked[0]
        if abs((rect.y0 + rect.y1) / 2 - center_y) > 300:
            continue
        key = (round(rect.x0, 1), round(rect.y0, 1),
               round(rect.x1, 1), round(rect.y1, 1))
        if key in used:
            continue
        used.add(key)
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"p{page_number:04d}-{len(used):02d}.png"
        target = target_dir / filename
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, alpha=False)
        target.write_bytes(pixmap.tobytes("png"))
        doc_dir = (media_dir.parent / Path(document.relative_path).parent)
        relative = os.path.relpath(target, doc_dir).replace("\\", "/")
        alt = caption.replace("]", "）").strip()
        output.append(("image", f"![{alt}]({relative})"))
        table = _ocr_table_markdown(target, ocr)
        if table:
            output.append(("table", table))
    return tuple(output)


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


def build_pdf_segments(
    page: fitz.Page,
    blocks: list[dict[str, Any]] | None = None,
) -> tuple[tuple[str, str], ...]:
    """按字体把页面拆成 code / paragraph 片段。

    等宽字体行归为 code，连续同类行合并；code 片段按花括号重建缩进。
    """
    if blocks is None:
        blocks = page.get_text("dict", flags=DICT_FLAGS).get("blocks", [])
    segments: list[list[Any]] = []
    for block in blocks:
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

    def _recognize(self, page: fitz.Page, dpi: int) -> list[Any]:
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
        return list(result or [])

    def extract(self, page: fitz.Page, dpi: int = 240) -> tuple[str, float]:
        result = self._recognize(page, dpi)
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

    def extract_items(
        self,
        page: fitz.Page,
        dpi: int = 240,
    ) -> list[tuple[tuple[float, float, float, float], str, float]]:
        """返回带坐标的识别结果（用于表格重建）。"""
        items: list[tuple[tuple[float, float, float, float], str, float]] = []
        for item in self._recognize(page, dpi):
            if len(item) < 3:
                continue
            box, text = item[0], str(item[1]).strip()
            score = float(item[2])
            if not text or score < self.min_confidence:
                continue
            try:
                xs = [float(point[0]) for point in box]
                ys = [float(point[1]) for point in box]
            except (TypeError, ValueError, IndexError):
                continue
            items.append(((min(xs), min(ys), max(xs), max(ys)), text, score))
        return items


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
    def __init__(
        self,
        ocr: OcrEngine | None = None,
        dpi: int = 240,
        media_dir: Path | None = None,
    ) -> None:
        self.ocr = ocr
        self.dpi = dpi
        self.media_dir = media_dir

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
                page_blocks = (
                    page.get_text("dict", flags=DICT_FLAGS).get("blocks", [])
                    if method == "pymupdf" else []
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
                        segments=(
                            (
                                *build_pdf_segments(page, page_blocks),
                                *(_image_markdown_for(
                                    document, page, index + 1, self.media_dir,
                                    page_blocks, self.ocr)
                                  if self.media_dir is not None else ()),
                            )
                            if method == "pymupdf"
                            else ()
                        ),
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
                    if kind in {"code", "table"}
                    else normalize_text(text)
                )
                if not normalized:
                    continue
                kind_value = {
                    "code": ElementKind.CODE,
                    "image": ElementKind.IMAGE,
                    "table": ElementKind.TABLE,
                }.get(kind, ElementKind.PARAGRAPH)
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
                        kind=kind_value,
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
