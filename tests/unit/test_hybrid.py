from __future__ import annotations

from secknow.retrieval.hybrid import query_channel_weights, weighted_rrf


def test_weighted_rrf_merges_channels_deterministically() -> None:
    fused = weighted_rrf(
        {
            "bm25": [("a", 8.0), ("b", 5.0)],
            "dense": [("b", 0.9), ("c", 0.8)],
        },
        {"bm25": 1.0, "dense": 1.0},
    )
    assert fused[0].item_id == "b"
    assert fused[0].ranks == {"bm25": 2, "dense": 1}


def test_exact_security_queries_prefer_lexical_channels() -> None:
    weights = query_channel_weights("CVE-2024-1234 rax")
    assert weights["bm25"] > weights["dense"]
    assert weights["sparse"] > weights["dense"]
