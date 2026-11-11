from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from secknow.domain.models import SourceRecord
from secknow.sources.adapters.base import (
    AdapterContext,
    BaseSourceAdapter,
    parse_datetime,
    read_json,
)


class AttackStixAdapter(BaseSourceAdapter):
    adapter_name = "attack_stix"
    suffixes = (".json",)

    def _objects(self, payload: object) -> list[dict[str, Any]]:
        if isinstance(payload, dict) and payload.get("type") == "bundle":
            objects = payload.get("objects")
            if not isinstance(objects, list):
                raise ValueError("STIX bundle objects must be a list")
            return [item for item in objects if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]
        raise ValueError("ATT&CK STIX must be an object or bundle")

    def iter_records(self, path: Path, context: AdapterContext) -> Iterator[SourceRecord]:
        for file in self.files(path):
            for item in self._objects(read_json(file)):
                stix_id = item.get("id")
                object_type = item.get("type")
                if not isinstance(stix_id, str) or not isinstance(object_type, str):
                    continue
                if object_type in {"relationship", "marking-definition"}:
                    continue
                external_references = item.get("external_references")
                if not isinstance(external_references, list):
                    external_references = []
                native_id = stix_id
                aliases: list[str] = []
                references: list[str] = []
                for reference in external_references:
                    if not isinstance(reference, dict):
                        continue
                    external_id = reference.get("external_id")
                    if isinstance(external_id, str):
                        if native_id == stix_id and external_id.startswith(("T", "S", "G", "M")):
                            native_id = external_id
                        else:
                            aliases.append(external_id)
                    if isinstance(reference.get("url"), str):
                        references.append(reference["url"])
                aliases.extend(
                    item.get(key, [])
                    for key in ("aliases", "x_mitre_aliases")
                    if isinstance(item.get(key), list)
                )
                flattened_aliases = [
                    alias
                    for value in aliases
                    for alias in (value if isinstance(value, list) else [value])
                    if isinstance(alias, str)
                ]
                yield self.record(
                    context=context,
                    path=file,
                    native_id=native_id,
                    title=str(item.get("name") or native_id),
                    body=str(item.get("description") or ""),
                    status="revoked"
                    if item.get("revoked")
                    else "deprecated"
                    if item.get("x_mitre_deprecated")
                    else "active",
                    aliases=(stix_id, *flattened_aliases),
                    references=references,
                    modified_at=parse_datetime(item.get("modified") or item.get("created")),
                    metadata={
                        "stix_id": stix_id,
                        "stix_type": object_type,
                        "kill_chain_phases": item.get("kill_chain_phases", []),
                        "platforms": item.get("x_mitre_platforms", []),
                        "permissions_required": item.get("x_mitre_permissions_required", []),
                    },
                )


ATTACKSTIXAdapter = AttackStixAdapter
