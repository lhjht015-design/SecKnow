import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from secknow.sources.adapters import (
    AdapterContext,
    AttackStixAdapter,
    CapaYamlAdapter,
    CapecXmlAdapter,
    CisaKevJsonAdapter,
    CveListV5JsonAdapter,
    CweXmlAdapter,
    GitMarkdownTreeAdapter,
    NvdJson20Adapter,
    OsvJsonAdapter,
    get_adapter,
)
from secknow.sources.identity import dedupe_records, record_identity
from secknow.sources.manifest import LicenseInfo


def context(source_id: str) -> AdapterContext:
    return AdapterContext(
        source_id=source_id,
        source_revision="fixture-1",
        license=LicenseInfo(
            expression="CC-BY-4.0",
            name="CC BY 4.0",
            url="https://creativecommons.org/licenses/by/4.0/",
            attribution_required=True,
            redistribution_allowed=True,
        ),
        snapshot_id="a" * 64,
        provenance_urls=("https://example.com/official",),
        observed_at=datetime(2026, 8, 5, tzinfo=UTC),
    )


def write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("adapter", "source_id", "payload", "expected"),
    [
        (
            CisaKevJsonAdapter(),
            "cisa-kev",
            {
                "catalogVersion": "2026.08.05",
                "vulnerabilities": [
                    {
                        "cveID": "CVE-2026-12345",
                        "vulnerabilityName": "Fixture vulnerability",
                        "shortDescription": "Exploited in the wild.",
                        "requiredAction": "Apply updates.",
                        "dateAdded": "2026-08-05",
                    }
                ],
            },
            "CVE-2026-12345",
        ),
        (
            CveListV5JsonAdapter(),
            "cve-list-v5",
            {
                "dataType": "CVE_RECORD",
                "dataVersion": "5.1",
                "cveMetadata": {"cveId": "CVE-2026-12345", "state": "PUBLISHED"},
                "containers": {
                    "cna": {
                        "title": "Fixture CVE",
                        "descriptions": [{"lang": "en", "value": "Description."}],
                        "references": [{"url": "https://example.com/advisory"}],
                    }
                },
            },
            "CVE-2026-12345",
        ),
        (
            NvdJson20Adapter(),
            "nvd",
            {
                "format": "NVD_CVE",
                "version": "2.0",
                "vulnerabilities": [
                    {
                        "cve": {
                            "id": "CVE-2026-12345",
                            "vulnStatus": "Analyzed",
                            "descriptions": [{"lang": "en", "value": "NVD description."}],
                        }
                    }
                ],
            },
            "CVE-2026-12345",
        ),
        (
            OsvJsonAdapter(),
            "osv",
            {
                "schema_version": "1.7.3",
                "id": "GHSA-2345-6789-CFGH",
                "modified": "2026-08-05T00:00:00Z",
                "summary": "OSV fixture",
                "details": "Affected package details.",
                "aliases": ["CVE-2026-12345"],
            },
            "GHSA-2345-6789-CFGH",
        ),
    ],
)
def test_json_adapters(
    tmp_path: Path, adapter: object, source_id: str, payload: object, expected: str
) -> None:
    path = write_json(tmp_path / "fixture.json", payload)
    records = adapter.load(path, context(source_id))  # type: ignore[attr-defined]

    assert len(records) == 1
    record = records[0]
    assert record.native_id == expected
    assert record.record_id == record_identity(source_id, expected)
    assert record.license == "CC-BY-4.0"
    assert record.metadata["snapshot_id"] == "a" * 64
    assert record.metadata["provenance_urls"] == ("https://example.com/official",)


def test_mitre_xml_adapters(tmp_path: Path) -> None:
    cwe = tmp_path / "cwe.xml"
    cwe.write_text(
        """<Weakness_Catalog xmlns="http://cwe.mitre.org/cwe-7">
<Weaknesses><Weakness ID="79" Name="XSS" Status="Stable">
<Description>Improper neutralization.</Description>
<References><Reference URL="https://cwe.mitre.org/data/definitions/79.html"/></References>
</Weakness></Weaknesses></Weakness_Catalog>""",
        encoding="utf-8",
    )
    capec = tmp_path / "capec.xml"
    capec.write_text(
        """<Attack_Pattern_Catalog xmlns="http://capec.mitre.org/capec-3">
<Attack_Patterns><Attack_Pattern ID="100" Name="Overflow" Status="Stable">
<Description>Overflow input.</Description></Attack_Pattern></Attack_Patterns>
</Attack_Pattern_Catalog>""",
        encoding="utf-8",
    )

    assert CweXmlAdapter().load(cwe, context("cwe"))[0].native_id == "CWE-79"
    assert CapecXmlAdapter().load(capec, context("capec"))[0].native_id == "CAPEC-100"


def test_attack_capa_and_markdown_adapters(tmp_path: Path) -> None:
    attack_path = write_json(
        tmp_path / "attack.json",
        {
            "type": "bundle",
            "objects": [
                {
                    "type": "attack-pattern",
                    "id": "attack-pattern--fixture",
                    "name": "Fixture technique",
                    "description": "Technique body.",
                    "external_references": [
                        {
                            "external_id": "T1234",
                            "url": "https://attack.mitre.org/techniques/T1234/",
                        }
                    ],
                }
            ],
        },
    )
    capa_path = tmp_path / "rule.yml"
    capa_path.write_text(
        """rule:
  meta:
    name: fixture capability
    namespace: persistence/service
    authors: [analyst]
  features:
    - or:
      - api: CreateService
""",
        encoding="utf-8",
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Debugging Guide\n\nNever execute samples.", encoding="utf-8")

    attack = AttackStixAdapter().load(attack_path, context("mitre-attack"))[0]
    capa = CapaYamlAdapter().load(capa_path, context("capa-rules"))[0]
    markdown = GitMarkdownTreeAdapter().load(docs, context("pwndbg-docs"))[0]

    assert attack.native_id == "T1234"
    assert attack.metadata["stix_id"] == "attack-pattern--fixture"
    assert capa.native_id == "persistence/service/fixture capability"
    assert markdown.native_id == "guide.md"
    assert markdown.title == "Debugging Guide"
    assert get_adapter("osv_json").adapter_name == "osv_json"


def test_conservative_dedupe_requires_same_source_id_and_hash(tmp_path: Path) -> None:
    path = write_json(
        tmp_path / "osv.json",
        {"id": "CVE-2026-12345", "summary": "same", "details": "same"},
    )
    adapter = OsvJsonAdapter()
    first = adapter.load(path, context("osv"))[0]
    duplicate = adapter.load(path, context("osv"))[0]
    other_source = adapter.load(path, context("mirror"))[0]

    assert dedupe_records((first, duplicate, other_source)) == (first, other_source)
