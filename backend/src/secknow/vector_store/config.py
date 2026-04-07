from __future__ import annotations

from datetime import datetime, timezone
from typing import Final

SUPPORTED_ZONES: Final[set[str]] = {"cyber", "ai", "crypto"}


def assert_zone(zone_id: str) -> None:
    if zone_id not in SUPPORTED_ZONES:
        raise ValueError(
            f"Unsupported zone_id='{zone_id}'. Expected one of: {sorted(SUPPORTED_ZONES)}"
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
