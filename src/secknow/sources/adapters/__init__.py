"""Local-file adapters for governed source snapshots."""

from secknow.sources.adapters.attack import ATTACKSTIXAdapter, AttackStixAdapter
from secknow.sources.adapters.base import AdapterContext, BaseSourceAdapter
from secknow.sources.adapters.capa import CAPAYAMLAdapter, CapaYamlAdapter
from secknow.sources.adapters.epss import EPSSCSVAdapter, EpssCsvAdapter
from secknow.sources.adapters.markdown import (
    GenericGitMarkdownTreeAdapter,
    GitMarkdownTreeAdapter,
)
from secknow.sources.adapters.mitre_xml import (
    CAPECXMLAdapter,
    CapecXmlAdapter,
    CWEXMLAdapter,
    CweXmlAdapter,
)
from secknow.sources.adapters.vulnerabilities import (
    CISAKEVJSONAdapter,
    CisaKevJsonAdapter,
    CVEListV5JSONAdapter,
    CveListV5JsonAdapter,
    NVDJSON20Adapter,
    NvdJson20Adapter,
    OSVJSONAdapter,
    OsvJsonAdapter,
)

_ADAPTERS: dict[str, type[BaseSourceAdapter]] = {
    "attack_stix": AttackStixAdapter,
    "capa_yaml": CapaYamlAdapter,
    "capec_xml": CapecXmlAdapter,
    "cisa_kev_json": CisaKevJsonAdapter,
    "cve_list_v5": CveListV5JsonAdapter,
    "epss_csv": EpssCsvAdapter,
    "git_markdown": GitMarkdownTreeAdapter,
    "nvd_json_20": NvdJson20Adapter,
    "osv_json": OsvJsonAdapter,
    "cwe_xml": CweXmlAdapter,
}


def get_adapter(name: str) -> BaseSourceAdapter:
    try:
        adapter = _ADAPTERS[name]
    except KeyError as exc:
        raise KeyError(f"unknown source adapter: {name}") from exc
    return adapter()


__all__ = [
    "ATTACKSTIXAdapter",
    "CAPAYAMLAdapter",
    "CAPECXMLAdapter",
    "CISAKEVJSONAdapter",
    "CVEListV5JSONAdapter",
    "CWEXMLAdapter",
    "EPSSCSVAdapter",
    "NVDJSON20Adapter",
    "OSVJSONAdapter",
    "AdapterContext",
    "AttackStixAdapter",
    "BaseSourceAdapter",
    "CapaYamlAdapter",
    "CapecXmlAdapter",
    "CisaKevJsonAdapter",
    "CveListV5JsonAdapter",
    "CweXmlAdapter",
    "EpssCsvAdapter",
    "GenericGitMarkdownTreeAdapter",
    "GitMarkdownTreeAdapter",
    "NvdJson20Adapter",
    "OsvJsonAdapter",
    "get_adapter",
]
