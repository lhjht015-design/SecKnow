from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from secknow.domain.models import Citation, SearchHit, SearchResponse
from secknow.retrieval.regroup import EvidenceCluster, regroup
from secknow.retrieval.reranker import RankedChunk


class EvidenceItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_id: str
    text: str
    score: float
    citation: Citation
    channels: dict[str, float] = Field(default_factory=dict)
    graph_paths: tuple[tuple[str, ...], ...] = ()
    explanation: str | None = None


class EvidenceSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    score: float
    items: tuple[EvidenceItem, ...]


class EvidenceAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    mode: str
    summary: str
    sections: tuple[EvidenceSection, ...]
    warnings: tuple[str, ...] = ()
    elapsed_ms: float
    generated: bool = False


class EvidenceBuilder:
    def build(self, response: SearchResponse) -> EvidenceAnswer:
        ranked = tuple(RankedChunk(hit.chunk, hit.score, hit.score) for hit in response.hits)
        graph_distances = {
            hit.chunk.chunk_id: min(len(path) - 1 for path in hit.graph_paths)
            for hit in response.hits
            if hit.graph_paths
        }
        clusters = regroup(ranked, graph_distances=graph_distances)
        lookup = {hit.chunk.chunk_id: hit for hit in response.hits}
        sections = tuple(self._section(cluster, lookup) for cluster in clusters)
        document_count = len(
            {item.citation.source_id for section in sections for item in section.items}
        )
        summary = (
            f"找到 {sum(len(section.items) for section in sections)} 条证据，"
            f"覆盖 {document_count} 个来源，归入 {len(sections)} 个主题。"
        )
        if not sections:
            summary = "未找到可引用的相关证据。请尝试更精确的术语、漏洞编号或函数名。"
        return EvidenceAnswer(
            query=response.query,
            mode=response.mode,
            summary=summary,
            sections=sections,
            warnings=response.warnings,
            elapsed_ms=response.elapsed_ms,
        )

    @staticmethod
    def _section(
        cluster: EvidenceCluster,
        lookup: dict[str, SearchHit],
    ) -> EvidenceSection:
        items: list[EvidenceItem] = []
        for ranked in cluster.chunks:
            hit = lookup[ranked.chunk.chunk_id]
            items.append(
                EvidenceItem(
                    evidence_id=ranked.chunk.chunk_id,
                    text=ranked.chunk.text,
                    score=ranked.score,
                    citation=ranked.chunk.citation,
                    channels=hit.channels,
                    graph_paths=hit.graph_paths,
                    explanation=hit.explanation,
                )
            )
        return EvidenceSection(
            title=cluster.label,
            score=cluster.score,
            items=tuple(items),
        )
