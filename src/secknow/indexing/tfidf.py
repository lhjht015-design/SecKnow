from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import jieba
import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from secknow.domain.ids import canonical_json_hash
from secknow.domain.models import Chunk


def security_tokens(text: str) -> list[str]:
    protected: list[str] = []
    current: list[str] = []
    for character in text:
        if character.isascii() and (character.isalnum() or character in "_./:+-#"):
            current.append(character.lower())
        else:
            if current:
                protected.append("".join(current))
                current = []
            if "㐀" <= character <= "鿿":
                protected.extend(token.strip() for token in jieba.cut(character) if token.strip())
    if current:
        protected.append("".join(current))
    chinese_text = "".join(character for character in text if "㐀" <= character <= "鿿")
    protected.extend(token.strip() for token in jieba.cut(chinese_text) if token.strip())
    return [token for token in protected if token]


@dataclass(frozen=True)
class TfidfHit:
    chunk_id: str
    score: float
    rank: int


class TfidfIndex:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            tokenizer=security_tokens,
            token_pattern=None,
            lowercase=False,
            sublinear_tf=True,
            norm="l2",
            max_df=1.0,
            min_df=1,
            dtype=np.float32,
        )
        self.matrix: sparse.csr_matrix | None = None
        self.chunk_ids: list[str] = []
        self.version: str | None = None

    def fit(self, chunks: Sequence[Chunk]) -> str:
        if not chunks:
            raise ValueError("Cannot fit an empty TF-IDF index")
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.matrix = self.vectorizer.fit_transform(chunk.text for chunk in chunks).tocsr()
        self.version = canonical_json_hash(
            {
                "chunk_ids": self.chunk_ids,
                "vocabulary": self.vectorizer.vocabulary_,
                "idf": self.vectorizer.idf_.round(7).tolist(),
            }
        )
        return self.version

    def search(self, query: str, top_k: int = 40) -> tuple[TfidfHit, ...]:
        if self.matrix is None:
            raise RuntimeError("TF-IDF index is not fitted")
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        if not np.any(scores > 0):
            return ()
        limit = min(top_k, len(scores))
        candidate_indices = np.argpartition(-scores, limit - 1)[:limit]
        ordered = candidate_indices[np.argsort(-scores[candidate_indices], kind="stable")]
        return tuple(
            TfidfHit(self.chunk_ids[index], float(scores[index]), rank)
            for rank, index in enumerate(ordered, 1)
            if scores[index] > 0
        )

    def save(self, directory: Path) -> Path:
        if self.matrix is None or self.version is None:
            raise RuntimeError("TF-IDF index is not fitted")
        target = directory / self.version
        temporary = directory / f".{self.version}.partial"
        temporary.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(temporary / "matrix.npz", self.matrix, compressed=True)
        joblib.dump(self.vectorizer, temporary / "vectorizer.joblib")
        (temporary / "manifest.json").write_text(
            json.dumps(
                {"version": self.version, "chunk_ids": self.chunk_ids},
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if target.exists():
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()
            return target
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, directory: Path) -> TfidfIndex:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        index = cls()
        index.matrix = sparse.load_npz(directory / "matrix.npz").tocsr()
        index.vectorizer = joblib.load(directory / "vectorizer.joblib")
        index.chunk_ids = list(manifest["chunk_ids"])
        index.version = str(manifest["version"])
        return index
