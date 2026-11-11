from __future__ import annotations

import re
from collections.abc import Iterable

from secknow.domain.ids import normalize_text, stable_id
from secknow.domain.models import SourceRecord

_CASE_INSENSITIVE_IDS = re.compile(
    r"^(?:CVE-\d{4}-\d{4,}|CWE-\d+|CAPEC-\d+|GHSA-[23456789CFGHJMPQRVWX]{4}"
    r"-[23456789CFGHJMPQRVWX]{4}-[23456789CFGHJMPQRVWX]{4})$",
    re.IGNORECASE,
)


def canonical_native_id(native_id: str) -> str:
    """Normalize structure without guessing that unrelated identifiers are equivalent."""
    value = normalize_text(native_id)
    if not value:
        raise ValueError("native source identifiers cannot be empty")
    if _CASE_INSENSITIVE_IDS.fullmatch(value):
        return value.upper()
    return value


def record_identity(source_id: str, native_id: str) -> str:
    return stable_id("source-record", source_id, canonical_native_id(native_id))


def exact_dedupe_key(record: SourceRecord) -> tuple[str, str, str]:
    return (
        record.source_id,
        canonical_native_id(record.native_id),
        record.normalized_sha256,
    )


def dedupe_records(records: Iterable[SourceRecord]) -> tuple[SourceRecord, ...]:
    """Remove only byte-equivalent normalized records from the same governed source."""
    unique: list[SourceRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        key = exact_dedupe_key(record)
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return tuple(unique)
