"""BM25 keyword index (Okapi BM25)."""

from __future__ import annotations

import math
from collections import Counter

from .text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.tfs: list[Counter] = []
        self.lengths: list[int] = []
        self.df: Counter = Counter()

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        tf = Counter(tokens)
        self.ids.append(doc_id)
        self.tfs.append(tf)
        self.lengths.append(len(tokens))
        self.df.update(tf.keys())

    def remove(self, doc_ids: set[str]) -> None:
        keep = [i for i, d in enumerate(self.ids) if d not in doc_ids]
        self.ids = [self.ids[i] for i in keep]
        self.tfs = [self.tfs[i] for i in keep]
        self.lengths = [self.lengths[i] for i in keep]
        self.df = Counter()
        for tf in self.tfs:
            self.df.update(tf.keys())

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        n = len(self.ids)
        if not n:
            return []
        avg = sum(self.lengths) / n or 1.0
        terms = tokenize(query)
        scores: list[tuple[str, float]] = []
        for doc_id, tf, length in zip(self.ids, self.tfs, self.lengths):
            s = 0.0
            for t in terms:
                if t not in tf:
                    continue
                idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                f = tf[t]
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / avg))
            if s > 0:
                scores.append((doc_id, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
