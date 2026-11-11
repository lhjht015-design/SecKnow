from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import yaml

from secknow.domain.models import SourceRecord
from secknow.sources.adapters.base import AdapterContext, BaseSourceAdapter


class CapaYamlAdapter(BaseSourceAdapter):
    adapter_name = "capa_yaml"
    suffixes = (".yaml", ".yml")

    def iter_records(self, path: Path, context: AdapterContext) -> Iterator[SourceRecord]:
        for file in self.files(path):
            with file.open("r", encoding="utf-8-sig") as stream:
                payload = yaml.safe_load(stream)
            if not isinstance(payload, dict) or not isinstance(payload.get("rule"), dict):
                raise ValueError(f"capa rule must contain a rule mapping: {file.name}")
            rule = payload["rule"]
            meta = rule.get("meta")
            if not isinstance(meta, dict) or not isinstance(meta.get("name"), str):
                raise ValueError(f"capa rule metadata must contain a name: {file.name}")
            name = meta["name"]
            namespace = meta.get("namespace")
            native_id = f"{namespace}/{name}" if isinstance(namespace, str) else name
            references = tuple(item for item in meta.get("references", []) if isinstance(item, str))
            examples = tuple(item for item in meta.get("examples", []) if isinstance(item, str))
            authors = tuple(item for item in meta.get("authors", []) if isinstance(item, str))
            description = (
                meta.get("description") if isinstance(meta.get("description"), str) else ""
            )
            yield self.record(
                context=context,
                path=file,
                native_id=native_id,
                title=name,
                body=description or name,
                references=references,
                metadata={
                    "namespace": namespace,
                    "authors": authors,
                    "scopes": meta.get("scopes", {}),
                    "examples": examples,
                    "rule_path": file.name,
                    "features": rule.get("features", []),
                },
            )


CAPAYAMLAdapter = CapaYamlAdapter
