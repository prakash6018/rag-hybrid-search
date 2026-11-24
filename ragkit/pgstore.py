"""PostgreSQL + pgvector store with the same interface as ``InMemoryStore``.

Vector search uses an HNSW index on the embeddings; keyword search uses
PostgreSQL full-text search (a generated tsvector column with a GIN index)
ranked with ts_rank_cd. Requires: psycopg[binary], pgvector.
"""

from __future__ import annotations

from .chunking import Chunk, content_hash
from .embeddings import Embedder

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id       text PRIMARY KEY,
    content_hash text NOT NULL,
    updated_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id  text PRIMARY KEY,
    doc_id    text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    idx       int  NOT NULL,
    heading   text NOT NULL,
    body      text NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', heading || ' ' || body)) STORED,
    embedding vector({dim}) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
"""


class PgVectorStore:
    def __init__(self, dsn: str, embedder: Embedder):
        import psycopg
        from pgvector.psycopg import register_vector

        self.embedder = embedder
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute(SCHEMA.format(dim=embedder.dim))
        register_vector(self.conn)

    def upsert_document(self, doc_id: str, text: str, chunks: list[Chunk]) -> bool:
        import numpy as np

        h = content_hash(text)
        row = self.conn.execute("SELECT content_hash FROM documents WHERE doc_id = %s", (doc_id,)).fetchone()
        if row and row[0] == h:
            return False
        vectors = self.embedder.embed([c.content for c in chunks])
        with self.conn.transaction():
            self.conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
            self.conn.execute("INSERT INTO documents (doc_id, content_hash) VALUES (%s, %s)", (doc_id, h))
            with self.conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (chunk_id, doc_id, idx, heading, body, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [
                        (c.id, doc_id, c.index, " > ".join(c.heading_path), c.text, np.array(v))
                        for c, v in zip(chunks, vectors)
                    ],
                )
        return True

    def delete_document(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))

    def vector_search(self, query_vec: list[float], k: int) -> list[tuple[str, float]]:
        import numpy as np

        rows = self.conn.execute(
            "SELECT chunk_id, 1 - (embedding <=> %s) AS similarity FROM chunks "
            "ORDER BY embedding <=> %s LIMIT %s",
            (np.array(query_vec), np.array(query_vec), k),
        ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def keyword_search(self, query: str, k: int) -> list[tuple[str, float]]:
        rows = self.conn.execute(
            "SELECT chunk_id, ts_rank_cd(tsv, q) AS rank "
            "FROM chunks, websearch_to_tsquery('english', %s) q "
            "WHERE tsv @@ q ORDER BY rank DESC LIMIT %s",
            (query, k),
        ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def get(self, chunk_id: str) -> Chunk:
        doc_id, idx, heading, body = self.conn.execute(
            "SELECT doc_id, idx, heading, body FROM chunks WHERE chunk_id = %s", (chunk_id,)
        ).fetchone()
        return Chunk(doc_id, idx, tuple(heading.split(" > ")) if heading else (), body)
