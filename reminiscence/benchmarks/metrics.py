"""Retrieval evaluation metrics: Recall@K, Precision@K, MRR."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & relevant_ids) / len(relevant_ids)


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    hits = sum(1 for i in top if i in relevant_ids)
    return hits / k


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    for pos, mid in enumerate(ranked_ids, start=1):
        if mid in relevant_ids:
            return 1.0 / pos
    return 0.0


def mrr(queries: Iterable[tuple[Sequence[str], set[str]]]) -> float:
    scores = [reciprocal_rank(r, rel) for r, rel in queries]
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(qrels: list[tuple[list[str], set[str]]], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """qrels: list of (ranked_ids, relevant_ids). Returns averaged metrics."""
    out: dict[str, float] = {}
    n = max(1, len(qrels))
    for k in ks:
        out[f"recall@{k}"] = round(sum(recall_at_k(r, rel, k) for r, rel in qrels) / n, 4)
        out[f"precision@{k}"] = round(sum(precision_at_k(r, rel, k) for r, rel in qrels) / n, 4)
    out["MRR"] = round(mrr((r, rel) for r, rel in qrels), 4)
    return out
