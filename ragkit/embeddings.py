"""Embedders.

``HashingEmbedder`` needs no model and no network: it hashes words and word
pairs into a fixed-size vector. It is far weaker than a real embedding model,
but it is deterministic, which makes the tests and the offline demo
reproducible. Swap in ``OpenAIEmbedder`` (or any class with the same two
methods) for real use.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence

from .text import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class HashingEmbedder:
    """Feature-hashing embedder over words and adjacent word pairs."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def _embed_one(self, text: str) -> list[float]:
        tokens = tokenize(text)
        vec = [0.0] * self.dim
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for f in features:
            i, sign = self._bucket(f)
            vec[i] += sign
        return _normalize(vec)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class OpenAIEmbedder:
    """Embeddings from the OpenAI API (or an Azure OpenAI deployment via base_url)."""

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536, **client_kwargs):
        from openai import OpenAI  # optional dependency

        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.dim = dim
        self.name = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=list(texts))
        return [d.embedding for d in resp.data]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
