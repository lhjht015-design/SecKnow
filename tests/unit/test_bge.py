from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from secknow.models.bge import BgeM3Encoder


class FakeBgeModel:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str], **_: Any) -> dict[str, object]:
        self.calls += 1
        return {
            "dense_vecs": np.asarray(
                [[float(len(text)), 1.0] for text in texts],
                dtype=np.float32,
            ),
            "lexical_weights": [{index: 0.5} for index, _ in enumerate(texts)],
        }


def test_bge_cache_reuses_content_without_loading_model(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    cache_path = tmp_path / "cache"
    model = FakeBgeModel()
    encoder = BgeM3Encoder(
        model_path,
        cache_dir=cache_path,
        model_revision="revision-a",
    )
    encoder._model = model

    dense, sparse = encoder.encode(["alpha", "beta"])
    assert model.calls == 1

    cached_encoder = BgeM3Encoder(
        model_path,
        cache_dir=cache_path,
        model_revision="revision-a",
    )
    cached_dense, cached_sparse = cached_encoder.encode(["alpha", "beta"])
    np.testing.assert_array_equal(cached_dense, dense)
    assert cached_sparse == sparse
    assert cached_encoder._model is None


def test_bge_cache_invalidates_when_unpinned_model_files_change(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    config = model_path / "config.json"
    config.write_text('{"version": 1}', encoding="utf-8")
    cache_path = tmp_path / "cache"
    first_model = FakeBgeModel()
    first = BgeM3Encoder(model_path, cache_dir=cache_path)
    first._model = first_model
    first.encode(["alpha"])
    assert first_model.calls == 1

    config.write_text('{"version": 200}', encoding="utf-8")
    second_model = FakeBgeModel()
    second = BgeM3Encoder(model_path, cache_dir=cache_path)
    second._model = second_model
    second.encode(["alpha"])
    assert second_model.calls == 1
