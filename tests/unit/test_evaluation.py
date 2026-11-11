from __future__ import annotations

import pytest

from secknow.evaluation import mean_metrics, retrieval_metrics


def test_retrieval_metrics_measure_rank_and_recall() -> None:
    metrics = retrieval_metrics({"a", "c"}, ["x", "a", "b", "c"], 4)
    assert metrics.precision_at_k == 0.5
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 0.5
    assert 0 < metrics.ndcg_at_k < 1


def test_mean_metrics_and_invalid_gold() -> None:
    first = retrieval_metrics({"a"}, ["a"], 1)
    second = retrieval_metrics({"b"}, ["x"], 1)
    aggregate = mean_metrics([first, second])
    assert aggregate.recall_at_k == 0.5
    with pytest.raises(ValueError, match="cannot be empty"):
        retrieval_metrics(set(), [], 1)
