from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from secknow.domain.ids import stable_id
from secknow.domain.models import Entity

CVE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
CWE = re.compile(r"\bCWE-\d+\b", re.IGNORECASE)
CAPEC = re.compile(r"\bCAPEC-\d+\b", re.IGNORECASE)
ATTACK = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
HEX = re.compile(r"\b0x[0-9a-fA-F]+\b")
FUNCTION = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\s*\(")
REGISTER = re.compile(
    r"\b(?:r(?:ax|bx|cx|dx|si|di|sp|bp|[89]|1[0-5])|e(?:ax|bx|cx|dx|si|di|sp|bp)|"
    r"[abcd][lh]|xmm\d+|ymm\d+)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EntityMention:
    entity: Entity
    start: int
    end: int
    surface: str


class EntityExtractor:
    def __init__(self, dictionaries: dict[str, set[str]] | None = None) -> None:
        self.dictionaries = dictionaries or {}

    @classmethod
    def from_yaml(cls, path: Path) -> EntityExtractor:
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        dictionaries = {
            str(kind): {str(term) for term in terms}
            for kind, terms in data.items()
            if isinstance(terms, list)
        }
        return cls(dictionaries)

    def extract(self, text: str) -> tuple[EntityMention, ...]:
        mentions: list[EntityMention] = []
        patterns = {
            "vulnerability": CVE,
            "weakness": CWE,
            "attack_pattern": CAPEC,
            "attack_technique": ATTACK,
            "address": HEX,
            "register": REGISTER,
        }
        for kind, pattern in patterns.items():
            for match in pattern.finditer(text):
                mentions.append(self._mention(kind, match.group(), match.start(), match.end()))
        for match in FUNCTION.finditer(text):
            surface = match.group().rstrip("(").strip()
            mentions.append(self._mention("code_symbol", surface, match.start(), match.end() - 1))
        folded = text.casefold()
        for kind, terms in self.dictionaries.items():
            for term in terms:
                start = 0
                while (index := folded.find(term.casefold(), start)) >= 0:
                    mentions.append(self._mention(kind, term, index, index + len(term)))
                    start = index + len(term)
        unique = {
            (mention.entity.entity_id, mention.start, mention.end): mention for mention in mentions
        }
        return tuple(sorted(unique.values(), key=lambda item: (item.start, item.end)))

    @staticmethod
    def _mention(kind: str, surface: str, start: int, end: int) -> EntityMention:
        canonical_kinds = {"vulnerability", "weakness", "attack_pattern"}
        canonical = surface.upper() if kind in canonical_kinds else surface.casefold()
        entity = Entity(
            entity_id=stable_id("entity", kind, canonical),
            canonical_name=canonical,
            kind=kind,
            aliases=(surface,) if surface != canonical else (),
        )
        return EntityMention(entity, start, end, surface)


def entity_document_frequency(
    documents: Iterable[tuple[str, Iterable[EntityMention]]],
) -> Counter[str]:
    frequency: Counter[str] = Counter()
    for _, mentions in documents:
        frequency.update({mention.entity.entity_id for mention in mentions})
    return frequency
