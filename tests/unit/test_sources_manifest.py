from pathlib import Path

import pytest
from pydantic import ValidationError

from secknow.sources.manifest import Manifest, RiskState, load_manifest

MANIFEST = Path(__file__).parents[2] / "sources" / "sources.yaml"


def test_default_manifest_is_strict_complete_and_governed() -> None:
    manifest = load_manifest(MANIFEST)

    assert manifest.version == 1
    assert {source.id for source in manifest.sources} == {
        "cve-list-v5",
        "nvd",
        "cisa-kev",
        "cwe",
        "capec",
        "mitre-attack",
        "epss",
        "osv",
        "rizin-book",
        "ghidra-docs",
        "pwndbg-docs",
        "capa-rules",
    }
    assert all(
        str(location.url).startswith("https://")
        for source in manifest.sources
        for location in source.locations
    )
    assert all(source.license.redistribution_allowed for source in manifest.sources)
    assert manifest.by_id("capa-rules").risk_state == RiskState.QUARANTINE
    assert "rizin-book" not in {source.id for source in manifest.enabled()}
    assert "rizin-book" in {source.id for source in manifest.enabled(include_opt_in=True)}
    assert "capa-rules" not in {source.id for source in manifest.enabled(include_opt_in=True)}


def test_manifest_forbids_unknown_fields_and_insecure_urls(tmp_path: Path) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text(
        """version: 1
sources:
  - id: sample
    name: Sample
    description: Sample source
    homepage: https://example.com
    adapter: osv_json
    risk_state: default
    unexpected: true
    license:
      expression: CC0-1.0
      name: CC0
      url: https://example.com/license
      attribution_required: false
      redistribution_allowed: true
    locations:
      - url: http://example.com/data.json
        kind: file
        media_type: application/json
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_manifest(path)


def test_manifest_model_is_frozen() -> None:
    manifest = load_manifest(MANIFEST)
    with pytest.raises(ValidationError):
        manifest.sources[0].name = "changed"  # type: ignore[misc]

    assert isinstance(manifest, Manifest)
