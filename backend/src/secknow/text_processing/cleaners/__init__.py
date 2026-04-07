from __future__ import annotations

"""清洗层入口。

导出 `clean_document_text`：流水线与门面 `load_document(normalize=True)` 的统一清洗入口。
"""

from .basic import basic_clean
from .document_clean import clean_document_text
from .normalize import normalize_for_file_type, normalize_text

__all__ = [
    "basic_clean",
    "clean_document_text",
    "normalize_for_file_type",
    "normalize_text",
]
