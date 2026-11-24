"""HTTP API.

    uvicorn api.main:app --reload

Set DATABASE_URL to use Postgres + pgvector (see docker-compose.yml);
otherwise everything is kept in memory. Set OPENAI_API_KEY to use real
embeddings; otherwise the offline HashingEmbedder is used.
"""

from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ragkit import HashingEmbedder, HybridRetriever, InMemoryStore, RetrievalProfile, chunk_markdown
from ragkit.evaluate import GoldenQuestion, evaluate


def _embedder():
    if os.getenv("OPENAI_API_KEY"):
        from ragkit.embeddings import OpenAIEmbedder

        return OpenAIEmbedder()
    return HashingEmbedder()


def _store(embedder):
    if dsn := os.getenv("DATABASE_URL"):
        from ragkit.pgstore import PgVectorStore

        return PgVectorStore(dsn, embedder)
    return InMemoryStore(embedder)


retriever = HybridRetriever(_store(_embedder()))
app = FastAPI(title="rag-hybrid-search", version="0.1.0")


class DocumentIn(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200)
    markdown: str = Field(min_length=1, max_length=500_000)
    max_words: int = Field(120, ge=20, le=1000)
    overlap_words: int = Field(20, ge=0, le=200)


class ProfileIn(BaseModel):
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid"
    fusion: Literal["rrf", "weighted"] = "rrf"
    alpha: float = Field(0.6, ge=0, le=1)
    candidates: int = Field(20, ge=1, le=200)
    min_score: float = Field(0.10, ge=0, le=1)
    top_k: int = Field(5, ge=1, le=50)
    context_budget_words: int = Field(600, ge=50, le=10_000)


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    profile: ProfileIn | None = None


class EvalIn(BaseModel):
    questions: list[GoldenQuestion]
    profile: ProfileIn | None = None


@app.get("/health")
def health():
    return {"ok": True, "embedder": retriever.store.embedder.name}


@app.post("/documents")
def upsert_document(doc: DocumentIn):
    if doc.overlap_words >= doc.max_words:
        raise HTTPException(422, "overlap_words must be smaller than max_words")
    chunks = chunk_markdown(doc.doc_id, doc.markdown, doc.max_words, doc.overlap_words)
    changed = retriever.store.upsert_document(doc.doc_id, doc.markdown, chunks)
    return {"doc_id": doc.doc_id, "chunks": len(chunks), "reindexed": changed}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    retriever.store.delete_document(doc_id)
    return {"deleted": doc_id}


@app.post("/search")
def search(body: SearchIn):
    profile = RetrievalProfile(**body.profile.model_dump()) if body.profile else None
    res = retriever.search(body.query, profile)
    return {
        "found": res.found,
        "reason": res.reason,
        "passages": [p.__dict__ for p in res.passages],
        # Ready to drop into an LLM prompt as reference text.
        "context": res.context(),
    }


@app.post("/evaluate")
def run_eval(body: EvalIn):
    profile = RetrievalProfile(**body.profile.model_dump()) if body.profile else None
    return evaluate(retriever, body.questions, profile).__dict__
