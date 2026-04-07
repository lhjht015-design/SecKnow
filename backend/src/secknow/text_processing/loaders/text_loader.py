from __future__ import annotations

"""通用纯文本加载器。"""

from pathlib import Path

import chardet

from ..exceptions import DocumentLoadError


def load_text(file_path: str | Path) -> str:
    """读取文本文件并自动尝试编码识别。"""
    path = Path(file_path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DocumentLoadError(f"读取文本文件失败: {path}") from exc

    detected = chardet.detect(raw)
    candidate_encodings = [
        (detected.get("encoding") or "").strip(),
        "utf-8",
        "gb18030",
        "latin-1",
    ]

    for enc in candidate_encodings:
        if not enc:
            continue
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue

    # 理论上 latin-1 可兜底，这里保留最后一道防线。
    return raw.decode("utf-8", errors="replace")
