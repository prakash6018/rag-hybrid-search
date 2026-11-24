"""Offline demo: index the sample docs, ask questions, run the golden-question eval.

    python -m scripts.demo
"""

import json
from pathlib import Path

from ragkit import HashingEmbedder, HybridRetriever, InMemoryStore, RetrievalProfile, chunk_markdown
from ragkit.evaluate import GoldenQuestion, evaluate

DATA = Path(__file__).resolve().parent.parent / "data"


def build() -> HybridRetriever:
    store = InMemoryStore(HashingEmbedder())
    for path in sorted((DATA / "docs").glob("*.md")):
        text = path.read_text()
        store.upsert_document(path.stem, text, chunk_markdown(path.stem, text))
    return HybridRetriever(store)


def main() -> None:
    retriever = build()
    print(f"Indexed {len(retriever.store.chunks)} chunks from {len(retriever.store.doc_hashes)} documents\n")

    for q in ["How long until I get my refund?", "Is a broken zip covered?", "What is the capital of France?"]:
        res = retriever.search(q)
        print(f"Q: {q}")
        if not res.found:
            print(f"   -> not found ({res.reason}). The assistant should say it doesn't know.\n")
            continue
        top = res.passages[0]
        print(f"   -> [{top.doc_id} / {top.heading}] score={top.score} sim={top.vector_similarity}")
        print(f"      {top.text[:110]}...\n")

    golden = [GoldenQuestion(**g) for g in json.loads((DATA / "golden.json").read_text())]
    print("Retrieval profiles compared on the golden questions:")
    for name, profile in {
        "hybrid (rrf)": RetrievalProfile(),
        "hybrid (weighted)": RetrievalProfile(fusion="weighted"),
        "vector only": RetrievalProfile(mode="vector"),
        "keyword only": RetrievalProfile(mode="keyword"),
        "hybrid, no min_score gate": RetrievalProfile(min_score=0.0),
    }.items():
        print(f"  {name:28s} {evaluate(retriever, golden, profile).summary()}")


if __name__ == "__main__":
    main()
