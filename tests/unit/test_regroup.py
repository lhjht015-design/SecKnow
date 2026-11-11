from __future__ import annotations

from secknow.domain.ids import sha256_bytes
from secknow.domain.models import Chunk, Citation
from secknow.retrieval.regroup import deduplicate, diversify, regroup
from secknow.retrieval.reranker import RankedChunk


def ranked_chunk(
    identifier: str,
    text: str,
    score: float,
    *,
    source_id: str = "guide",
    page: int = 1,
    order: int = 0,
) -> RankedChunk:
    chunk = Chunk(
        chunk_id=identifier,
        source_id=source_id,
        source_version="v1",
        text=text,
        content_sha256=sha256_bytes(text.encode()),
        element_ids=(f"element:{identifier}",),
        order=order,
        token_count=max(1, len(text.split())),
        section_path=("Memory corruption",),
        citation=Citation(
            source_id=source_id,
            source_version="v1",
            title="Security guide",
            page_start=page,
        ),
    )
    return RankedChunk(chunk=chunk, score=score, original_score=score)


def test_deduplicate_keeps_best_exact_and_near_duplicate() -> None:
    ranked = (
        ranked_chunk("low", "ROP chains reuse short instruction sequences.", 0.5),
        ranked_chunk("best", "ROP chains reuse short instruction sequences.", 0.9),
        ranked_chunk("near", "ROP chains reuse short instruction sequences!", 0.8),
        ranked_chunk("other", "Heap metadata corruption changes allocator state.", 0.7),
    )

    unique = deduplicate(ranked)

    assert [item.chunk.chunk_id for item in unique] == ["best", "other"]


def test_diversify_uses_mmr_to_surface_distinct_evidence() -> None:
    ranked = (
        ranked_chunk("rop-1", "ROP gadgets build a chain that controls execution flow.", 1.0),
        ranked_chunk("rop-2", "ROP gadget chains control program execution flow.", 0.99),
        ranked_chunk("heap", "Heap bins and allocator metadata govern free chunk reuse.", 0.82),
    )

    diversified = diversify(ranked, relevance_weight=0.55, limit=2)

    assert [item.chunk.chunk_id for item in diversified] == ["rop-1", "heap"]


def test_regroup_orders_related_chunks_by_graph_distance_and_source_position() -> None:
    ranked = (
        ranked_chunk(
            "page-3", "CVE parser memory corruption permits code execution.", 0.90, page=3, order=3
        ),
        ranked_chunk(
            "page-1", "CVE parser memory corruption causes code execution.", 0.88, page=1, order=1
        ),
        ranked_chunk(
            "page-2",
            "CVE parser memory corruption may enable code execution.",
            0.87,
            page=2,
            order=2,
        ),
    )

    clusters = regroup(
        ranked,
        graph_distances={"page-3": 2, "page-1": 1, "page-2": 1},
    )

    assert len(clusters) == 1
    assert [item.chunk.chunk_id for item in clusters[0].chunks] == [
        "page-1",
        "page-2",
        "page-3",
    ]
