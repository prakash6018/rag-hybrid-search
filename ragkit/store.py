"""In-memory vector + keyword store. ``pgstore.PgVectorStore`` has the same interface."""

from __future__ import annotations

from .chunking import Chunk, content_hash
from .embeddings import Embedder, cosine
from .keyword import BM25Index


class InMemoryStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: dict[str, Chunk] = {}
        self.vectors: dict[str, list[float]] = {}
        self.doc_hashes: dict[str, str] = {}
        self.bm25 = BM25Index()

    def upsert_document(self, doc_id: str, text: str, chunks: list[Chunk]) -> bool:
        """Index a document. Returns False (and does nothing) if it hasn't changed."""
        h = content_hash(text)
        if self.doc_hashes.get(doc_id) == h:
            return False
        self.delete_document(doc_id)
        vectors = self.embedder.embed([c.content for c in chunks])
        for c, v in zip(chunks, vectors):
            self.chunks[c.id] = c
            self.vectors[c.id] = v
            self.bm25.add(c.id, c.content)
        self.doc_hashes[doc_id] = h
        return True

    def delete_document(self, doc_id: str) -> None:
        stale = {cid for cid, c in self.chunks.items() if c.doc_id == doc_id}
        for cid in stale:
            self.chunks.pop(cid)
            self.vectors.pop(cid)
        self.bm25.remove(stale)
        self.doc_hashes.pop(doc_id, None)

    def vector_search(self, query_vec: list[float], k: int) -> list[tuple[str, float]]:
        scored = [(cid, cosine(query_vec, v)) for cid, v in self.vectors.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def keyword_search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.bm25.search(query, k)

    def get(self, chunk_id: str) -> Chunk:
        return self.chunks[chunk_id]
