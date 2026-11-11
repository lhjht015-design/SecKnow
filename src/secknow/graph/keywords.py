from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx

from secknow.indexing.tfidf import security_tokens

_SECURITY_IDENTIFIER = re.compile(
    r"^(?:cve-\d{4}-\d{4,7}|cwe-\d+|capec-\d+|t\d{4}(?:\.\d{3})?|0x[0-9a-f]+)$"
)
_STOP_WORDS = {
    "and",
    "are",
    "for",
    "from",
    "into",
    "that",
    "the",
    "this",
    "with",
    "一个",
    "以及",
    "可以",
    "使用",
    "进行",
    "通过",
    "这个",
}


@dataclass(frozen=True)
class KeywordScore:
    token: str
    score: float
    idf: float
    textrank: float


def extract_keywords(
    texts: Sequence[str],
    *,
    top_k: int = 6,
    window_size: int = 4,
) -> tuple[tuple[KeywordScore, ...], ...]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if window_size < 2:
        raise ValueError("window_size must be at least two")
    tokenized = [tuple(_tokens(text)) for text in texts]
    document_frequency = Counter(token for tokens in tokenized for token in set(tokens))
    document_count = max(1, len(texts))
    results: list[tuple[KeywordScore, ...]] = []
    for tokens in tokenized:
        if not tokens:
            results.append(())
            continue
        frequency = Counter(tokens)
        graph = _cooccurrence_graph(tokens, window_size)
        ranks = nx.pagerank(graph, alpha=0.85, weight="weight") if graph else {}
        raw: list[KeywordScore] = []
        for token, count in frequency.items():
            ratio = document_frequency[token] / document_count
            if document_count >= 5 and ratio > 0.85 and not _SECURITY_IDENTIFIER.fullmatch(token):
                continue
            idf = math.log((1 + document_count) / (1 + document_frequency[token])) + 1
            textrank = float(ranks.get(token, 0.0))
            tfidf = (count / len(tokens)) * idf
            raw.append(KeywordScore(token, tfidf, idf, textrank))
        max_tfidf = max((item.score for item in raw), default=1.0)
        max_textrank = max((item.textrank for item in raw), default=1.0)
        combined = tuple(
            KeywordScore(
                token=item.token,
                score=0.45 * item.score / max_tfidf + 0.55 * item.textrank / max_textrank,
                idf=item.idf,
                textrank=item.textrank,
            )
            for item in raw
        )
        results.append(
            tuple(sorted(combined, key=lambda item: (-item.score, -item.idf, item.token))[:top_k])
        )
    return tuple(results)


def _tokens(text: str) -> list[str]:
    return [
        token
        for raw in security_tokens(text)
        if (token := raw.casefold().strip())
        and token not in _STOP_WORDS
        and (len(token) > 1 or _SECURITY_IDENTIFIER.fullmatch(token))
    ]


def _cooccurrence_graph(tokens: Sequence[str], window_size: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(tokens)
    for index, source in enumerate(tokens):
        for target in tokens[index + 1 : index + window_size]:
            if source == target:
                continue
            weight = float(graph.get_edge_data(source, target, {}).get("weight", 0.0)) + 1.0
            graph.add_edge(source, target, weight=weight)
    return graph
