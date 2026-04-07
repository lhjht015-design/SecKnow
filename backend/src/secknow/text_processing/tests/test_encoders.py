from __future__ import annotations

import secknow.text_processing.encoders.sbert as sbert_module
from secknow.text_processing.encoders.sbert import (
    DeterministicFakeEmbedder,
    SBERTEmbedder,
)


def test_fake_embedder_dim_384() -> None:
    """fake 编码器默认返回 384 维向量。"""
    embedder = DeterministicFakeEmbedder(dim=384)
    vectors = embedder.encode(["alpha", "beta"])
    assert len(vectors) == 2
    assert all(len(vec) == 384 for vec in vectors)


def test_sbert_embedder_dim_384(monkeypatch) -> None:
    """sbert 编码器应输出 384 维（通过 monkeypatch 避免下载模型）。"""

    class _FakeSentenceTransformer:
        def __init__(self, _model_name: str):
            self.model_name = _model_name

        def encode(self, texts, **_kwargs):
            return [[0.1] * 384 for _ in texts]

    monkeypatch.setattr(sbert_module, "_SentenceTransformer", _FakeSentenceTransformer)

    embedder = SBERTEmbedder(model_name="dummy", dim=384)
    vectors = embedder.encode(["测试文本"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 384
