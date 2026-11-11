from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

from secknow.domain.models import SourceRecord
from secknow.sources.adapters.base import AdapterContext, BaseSourceAdapter


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> str:
    for child in element:
        if _local_name(child.tag) == name:
            return " ".join("".join(child.itertext()).split())
    return ""


def _references(element: ET.Element) -> tuple[str, ...]:
    values: list[str] = []
    for child in element.iter():
        if _local_name(child.tag) == "Reference":
            url = child.attrib.get("URL")
            if url:
                values.append(url)
    return tuple(values)


class _MitreCatalogXmlAdapter(BaseSourceAdapter):
    element_name: str
    id_prefix: str

    def iter_records(self, path: Path, context: AdapterContext) -> Iterator[SourceRecord]:
        for file in self.files(path):
            for _event, element in ET.iterparse(file, events=("end",)):
                if _local_name(element.tag) != self.element_name:
                    continue
                raw_id = element.attrib.get("ID")
                if not raw_id:
                    element.clear()
                    continue
                native_id = f"{self.id_prefix}-{raw_id}"
                title = element.attrib.get("Name") or native_id
                description = _child_text(element, "Description")
                extended = _child_text(element, "Extended_Description")
                consequences = [
                    " ".join("".join(child.itertext()).split())
                    for child in element.iter()
                    if _local_name(child.tag) == "Consequence"
                ]
                body = "\n\n".join(
                    value for value in (description, extended, *consequences) if value
                )
                related: list[str] = []
                for child in element.iter():
                    related_id = child.attrib.get(f"{self.id_prefix}_ID")
                    if related_id:
                        related.append(f"{self.id_prefix}-{related_id}")
                yield self.record(
                    context=context,
                    path=file,
                    native_id=native_id,
                    title=title,
                    body=body,
                    status=element.attrib.get("Status", "active").lower(),
                    references=_references(element),
                    metadata={
                        "abstraction": element.attrib.get("Abstraction"),
                        "structure": element.attrib.get("Structure"),
                        "related_ids": related,
                    },
                )
                element.clear()


class CweXmlAdapter(_MitreCatalogXmlAdapter):
    adapter_name = "cwe_xml"
    suffixes = (".xml",)
    element_name = "Weakness"
    id_prefix = "CWE"


class CapecXmlAdapter(_MitreCatalogXmlAdapter):
    adapter_name = "capec_xml"
    suffixes = (".xml",)
    element_name = "Attack_Pattern"
    id_prefix = "CAPEC"


CWEXMLAdapter = CweXmlAdapter
CAPECXMLAdapter = CapecXmlAdapter
