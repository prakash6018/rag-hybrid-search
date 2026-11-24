"""Golden-question evaluation: does retrieval find the passage it should?"""

from __future__ import annotations

from dataclasses import dataclass

from .retriever import HybridRetriever, RetrievalProfile


@dataclass
class GoldenQuestion:
    question: str
    expected_doc: str | None   # None = out of scope, the retriever should find nothing
    expected_heading: str | None = None


@dataclass
class EvalReport:
    recall_at_k: float
    precision_at_3: float
    mrr: float
    abstain_accuracy: float
    answerable: int
    unanswerable: int
    failures: list[str]

    def summary(self) -> str:
        return (
            f"recall@k={self.recall_at_k:.2f}  precision@3={self.precision_at_3:.2f}  "
            f"MRR={self.mrr:.2f}  abstain_accuracy={self.abstain_accuracy:.2f}  "
            f"({self.answerable} answerable, {self.unanswerable} out of scope)"
        )


def _hit(passage, q: GoldenQuestion) -> bool:
    if passage.doc_id != q.expected_doc:
        return False
    return q.expected_heading is None or q.expected_heading.lower() in passage.heading.lower()


def evaluate(
    retriever: HybridRetriever, questions: list[GoldenQuestion], profile: RetrievalProfile | None = None
) -> EvalReport:
    answerable = [q for q in questions if q.expected_doc]
    unanswerable = [q for q in questions if not q.expected_doc]
    recall_hits = precision_sum = rr_sum = 0.0
    failures: list[str] = []

    for q in answerable:
        res = retriever.search(q.question, profile)
        hits = [_hit(p, q) for p in res.passages]
        if any(hits):
            recall_hits += 1
            rr_sum += 1 / (hits.index(True) + 1)
        else:
            failures.append(f"missed: {q.question!r}")
        top3 = hits[:3]
        precision_sum += (sum(top3) / len(top3)) if top3 else 0.0

    abstained = 0
    for q in unanswerable:
        res = retriever.search(q.question, profile)
        if not res.found:
            abstained += 1
        else:
            failures.append(f"answered out-of-scope question: {q.question!r}")

    n = len(answerable) or 1
    return EvalReport(
        recall_at_k=recall_hits / n,
        precision_at_3=precision_sum / n,
        mrr=rr_sum / n,
        abstain_accuracy=(abstained / len(unanswerable)) if unanswerable else 1.0,
        answerable=len(answerable),
        unanswerable=len(unanswerable),
        failures=failures,
    )
