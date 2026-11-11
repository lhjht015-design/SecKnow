from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from secknow.domain.models import SourceRecord
from secknow.sources.adapters.base import (
    AdapterContext,
    BaseSourceAdapter,
    parse_datetime,
    read_json,
    text_values,
)


def _object(value: object) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _urls(values: object) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    return tuple(
        item["url"]
        for item in values
        if isinstance(item, dict) and isinstance(item.get("url"), str)
    )


def _description(values: object) -> str:
    if not isinstance(values, list):
        return ""
    english: list[str] = []
    fallback: list[str] = []
    for item in values:
        if not isinstance(item, dict) or not isinstance(item.get("value"), str):
            continue
        fallback.append(item["value"])
        if item.get("lang") in {"en", "en-US"}:
            english.append(item["value"])
    selected = english or fallback
    return "\n\n".join(selected)


class CisaKevJsonAdapter(BaseSourceAdapter):
    adapter_name = "cisa_kev_json"
    suffixes = (".json",)

    def iter_records(self, path: Path, context: AdapterContext) -> Iterator[SourceRecord]:
        for file in self.files(path):
            payload = read_json(file)
            if not isinstance(payload, dict) or not isinstance(
                payload.get("vulnerabilities"), list
            ):
                raise ValueError("CISA KEV JSON must contain a vulnerabilities list")
            catalog_version = str(payload.get("catalogVersion", context.source_revision))
            for item in payload["vulnerabilities"]:
                if not isinstance(item, dict) or not isinstance(item.get("cveID"), str):
                    continue
                cve_id = item["cveID"]
                title = str(item.get("vulnerabilityName") or cve_id)
                sections = [
                    str(item.get("shortDescription") or ""),
                    (
                        f"Required action: {item['requiredAction']}"
                        if item.get("requiredAction")
                        else ""
                    ),
                    f"Known ransomware use: {item['knownRansomwareCampaignUse']}"
                    if item.get("knownRansomwareCampaignUse")
                    else "",
                    f"Notes: {item['notes']}" if item.get("notes") else "",
                ]
                yield self.record(
                    context=context,
                    path=file,
                    native_id=cve_id,
                    title=title,
                    body="\n\n".join(section for section in sections if section),
                    references=(
                        f"https://www.cve.org/CVERecord?id={cve_id}",
                        *([item["notes"]] if str(item.get("notes", "")).startswith("http") else []),
                    ),
                    modified_at=parse_datetime(item.get("dateAdded")),
                    metadata={
                        "catalog_version": catalog_version,
                        "vendor_project": item.get("vendorProject"),
                        "product": item.get("product"),
                        "date_added": item.get("dateAdded"),
                        "due_date": item.get("dueDate"),
                        "cwes": item.get("cwes", []),
                    },
                )


class CveListV5JsonAdapter(BaseSourceAdapter):
    adapter_name = "cve_list_v5"
    suffixes = (".json",)

    def iter_records(self, path: Path, context: AdapterContext) -> Iterator[SourceRecord]:
        for file in self.files(path):
            payload = read_json(file)
            if not isinstance(payload, dict):
                raise ValueError("CVE List V5 record must be a JSON object")
            metadata = payload.get("cveMetadata")
            containers = payload.get("containers")
            if not isinstance(metadata, dict) or not isinstance(containers, dict):
                raise ValueError("CVE List V5 record lacks cveMetadata or containers")
            cve_id = metadata.get("cveId")
            if not isinstance(cve_id, str):
                raise ValueError("CVE List V5 cveMetadata.cveId must be a string")
            cna = _object(containers.get("cna"))
            adp = _list(containers.get("adp"))
            descriptions = _description(cna.get("descriptions"))
            problem_types = _list(cna.get("problemTypes"))
            cwes: list[str] = []
            for problem in problem_types:
                if isinstance(problem, dict):
                    for description in _list(problem.get("descriptions")):
                        if isinstance(description, dict) and isinstance(
                            description.get("cweId"), str
                        ):
                            cwes.append(description["cweId"])
            affected = _list(cna.get("affected"))
            products = [
                f"{item.get('vendor', '')} {item.get('product', '')}".strip()
                for item in affected
                if isinstance(item, dict)
            ]
            references = list(_urls(cna.get("references")))
            for enrichment in adp:
                if isinstance(enrichment, dict):
                    references.extend(_urls(enrichment.get("references")))
            raw_title = cna.get("title")
            title = raw_title if isinstance(raw_title, str) else cve_id
            yield self.record(
                context=context,
                path=file,
                native_id=cve_id,
                title=title,
                body=descriptions,
                status=str(metadata.get("state", "PUBLISHED")).lower(),
                references=references,
                modified_at=parse_datetime(
                    metadata.get("dateUpdated") or metadata.get("datePublished")
                ),
                metadata={
                    "data_type": payload.get("dataType"),
                    "data_version": payload.get("dataVersion"),
                    "assigner": metadata.get("assignerShortName"),
                    "products": products,
                    "cwes": cwes,
                },
            )


class NvdJson20Adapter(BaseSourceAdapter):
    adapter_name = "nvd_json_20"
    suffixes = (".json",)

    def iter_records(self, path: Path, context: AdapterContext) -> Iterator[SourceRecord]:
        for file in self.files(path):
            payload = read_json(file)
            if not isinstance(payload, dict) or payload.get("format") not in {None, "NVD_CVE"}:
                raise ValueError("invalid NVD JSON 2.0 payload")
            vulnerabilities = payload.get("vulnerabilities")
            if not isinstance(vulnerabilities, list):
                raise ValueError("NVD JSON 2.0 must contain a vulnerabilities list")
            for wrapper in vulnerabilities:
                if not isinstance(wrapper, dict) or not isinstance(wrapper.get("cve"), dict):
                    continue
                cve: dict[str, Any] = wrapper["cve"]
                native_id = cve.get("id")
                if not isinstance(native_id, str):
                    continue
                weaknesses = _list(cve.get("weaknesses"))
                cwes: list[str] = []
                for weakness in weaknesses:
                    if isinstance(weakness, dict):
                        cwes.extend(text_values(weakness.get("description")))
                references = _urls(cve.get("references"))
                yield self.record(
                    context=context,
                    path=file,
                    native_id=native_id,
                    title=native_id,
                    body=_description(cve.get("descriptions")),
                    status=str(cve.get("vulnStatus", "active")).lower(),
                    aliases=tuple(cve.get("cveTags", []))
                    if isinstance(cve.get("cveTags"), list)
                    else (),
                    references=references,
                    modified_at=parse_datetime(cve.get("lastModified") or cve.get("published")),
                    metadata={
                        "published": cve.get("published"),
                        "cwes": cwes,
                        "metrics": cve.get("metrics", {}),
                        "configurations": cve.get("configurations", []),
                    },
                )


class OsvJsonAdapter(BaseSourceAdapter):
    adapter_name = "osv_json"
    suffixes = (".json",)

    def _documents(self, payload: object) -> list[dict[str, Any]]:
        if isinstance(payload, dict) and isinstance(payload.get("vulns"), list):
            return [item for item in payload["vulns"] if isinstance(item, dict)]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]
        raise ValueError("OSV JSON must be a vulnerability, array, or vulns envelope")

    def iter_records(self, path: Path, context: AdapterContext) -> Iterator[SourceRecord]:
        for file in self.files(path):
            for item in self._documents(read_json(file)):
                native_id = item.get("id")
                if not isinstance(native_id, str):
                    continue
                aliases = tuple(
                    alias
                    for alias in _list(item.get("aliases"))
                    if isinstance(alias, str)
                )
                references = tuple(
                    reference["url"]
                    for reference in _list(item.get("references"))
                    if isinstance(reference, dict) and isinstance(reference.get("url"), str)
                )
                affected = _list(item.get("affected"))
                packages = [
                    affected_item.get("package")
                    for affected_item in affected
                    if isinstance(affected_item, dict)
                    and isinstance(affected_item.get("package"), dict)
                ]
                severity = _list(item.get("severity"))
                yield self.record(
                    context=context,
                    path=file,
                    native_id=native_id,
                    title=str(item.get("summary") or native_id),
                    body=str(item.get("details") or item.get("summary") or ""),
                    status="withdrawn" if item.get("withdrawn") else "active",
                    aliases=aliases,
                    references=references,
                    modified_at=parse_datetime(item.get("modified") or item.get("published")),
                    metadata={
                        "schema_version": item.get("schema_version"),
                        "published": item.get("published"),
                        "withdrawn": item.get("withdrawn"),
                        "packages": packages,
                        "severity": severity,
                        "database_specific": item.get("database_specific", {}),
                    },
                )


CISAKEVJSONAdapter = CisaKevJsonAdapter
CVEListV5JSONAdapter = CveListV5JsonAdapter
NVDJSON20Adapter = NvdJson20Adapter
OSVJSONAdapter = OsvJsonAdapter
