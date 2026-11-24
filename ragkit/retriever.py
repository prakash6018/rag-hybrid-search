"""Hybrid retrieval with a minimum-similarity gate.

The gate is the part that matters in production: if no passage is actually
similar to the question, the retriever returns ``found=False`` and the
assistant says it doesn't know, instead of answering from the "least bad"
chunk. Rank fusion alone can't do this, because it always gives the top
result a high score no matter how irrelevant it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .fusion import reciprocal_rank_fusion, weighted_fusion
from .store import InMemoryStore


@dataclass(frozen=True)
class RetrievalProfile:
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid"
    candidates: int = 20          # taken from each of the vector and keyword lists
    fusion: Literal["rrf", "weighted"] = "rrf"
    alpha: float = 0.6            # weight of the vector side in weighted fusion
    # Minimum cosine similarity for a vector candidate. This depends on the
    # embedding model: 0.10 suits the offline HashingEmbedder (measured on the
    # golden questions); real models need their own value, e.g. ~0.30 for
    # text-embedding-3-small. Re-run the eval when you change models.
    min_score: float = 0.10
    top_k: int = 5
    context_budget_words: int = 600


@dataclass
class Passage:
    chunk_id: str
    doc_id: str
    heading: str
    text: str
    score: float
    vector_similarity: float | None


@dataclass
class SearchResult:
    query: str
    found: bool
    passages: list[Passage] = field(default_factory=list)
    reason: str = ""

    def context(self) -> str:
        """Passages formatted as reference text for an LLM prompt."""
        return "\n\n".join(
            f"[{i}] ({p.doc_id} / {p.heading})\n{p.text}" for i, p in enumerate(self.passages, 1)
        )


class HybridRetriever:
    def __init__(self, store: InMemoryStore, profile: RetrievalProfile | None = None):
        self.store = store
        self.profile = profile or RetrievalProfile()

    def search(self, query: str, profile: RetrievalProfile | None = None) -> SearchResult:
        p = profile or self.profile
        qvec = self.store.embedder.embed([query])[0]

        vec_hits: list[tuple[str, float]] = []
        if p.mode in ("hybrid", "vector"):
            # The gate: a vector candidate must itself be similar enough.
            vec_hits = [h for h in self.store.vector_search(qvec, p.candidates) if h[1] >= p.min_score]
        kw_hits: list[tuple[str, float]] = []
        if p.mode in ("hybrid", "keyword"):
            kw_hits = self.store.keyword_search(query, p.candidates)
            if p.mode == "hybrid":
                # In hybrid mode a keyword-only hit is kept only if the vector side
                # found *something* relevant; otherwise a stray shared word could
                # make an unrelated question look answerable.
                if not vec_hits:
                    kw_hits = []

        if not vec_hits and not kw_hits:
            return SearchResult(query, False, reason=f"no passage reached min_score={p.min_score}")

        if p.mode == "vector":
            fused = dict(vec_hits)
        elif p.mode == "keyword":
            fused = dict(kw_hits)
        elif p.fusion == "rrf":
            fused = reciprocal_rank_fusion([[c for c, _ in vec_hits], [c for c, _ in kw_hits]])
        else:
            fused = weighted_fusion(dict(vec_hits), dict(kw_hits), p.alpha)

        sims = dict(vec_hits)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: p.top_k]
        passages, used = [], 0
        for cid, score in ranked:
            c = self.store.get(cid)
            words = len(c.text.split())
            if passages and used + words > p.context_budget_words:
                break
            used += words
            passages.append(
                Passage(cid, c.doc_id, " > ".join(c.heading_path), c.text, round(score, 4),
                        round(sims[cid], 4) if cid in sims else None)
            )
        return SearchResult(query, True, passages)
