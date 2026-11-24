import json
from pathlib import Path

from ragkit import HashingEmbedder, HybridRetriever, InMemoryStore, RetrievalProfile, chunk_markdown
from ragkit.evaluate import GoldenQuestion, evaluate
from ragkit.fusion import reciprocal_rank_fusion
from ragkit.keyword import BM25Index

DATA = Path(__file__).resolve().parent.parent / "data"


def build():
    store = InMemoryStore(HashingEmbedder())
    for p in sorted((DATA / "docs").glob("*.md")):
        store.upsert_document(p.stem, p.read_text(), chunk_markdown(p.stem, p.read_text()))
    return HybridRetriever(store)


def golden():
    return [GoldenQuestion(**g) for g in json.loads((DATA / "golden.json").read_text())]


def test_finds_the_right_section():
    res = build().search("How long until I get my refund?")
    assert res.found
    assert res.passages[0].doc_id == "returns"
    assert "Refunds" in res.passages[0].heading


def test_says_not_found_for_out_of_scope_questions():
    res = build().search("What is the capital of France?")
    assert not res.found and res.passages == []


def test_without_the_gate_it_answers_anything():
    res = build().search("What is the capital of France?", RetrievalProfile(min_score=0.0))
    assert res.found  # this is the failure the min_score gate exists to prevent


def test_golden_questions_meet_the_bar():
    report = evaluate(build(), golden())
    assert report.recall_at_k == 1.0
    assert report.mrr >= 0.9
    assert report.abstain_accuracy == 1.0, report.failures


def test_unchanged_documents_are_not_reindexed():
    r = build()
    text = (DATA / "docs" / "returns.md").read_text()
    assert r.store.upsert_document("returns", text, chunk_markdown("returns", text)) is False


def test_rrf_is_normalised_and_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
    assert fused["a"] == 1.0
    assert fused["b"] < fused["a"] and fused["c"] < fused["a"]


def test_bm25_ranks_the_matching_document_first():
    idx = BM25Index()
    idx.add("1", "refunds go back to the original payment method")
    idx.add("2", "standard delivery takes three days")
    assert idx.search("refund payment", 2)[0][0] == "1"


def test_context_budget_limits_the_prompt():
    res = build().search("returns refunds exchanges", RetrievalProfile(top_k=10, context_budget_words=40))
    assert sum(len(p.text.split()) for p in res.passages[1:]) <= 40
