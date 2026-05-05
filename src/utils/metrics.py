"""Evaluation metrics for retrieval: nDCG@k, MRR, Recall@k."""
from __future__ import annotations

import math
from typing import Sequence


def dcg(relevances: Sequence[int]) -> float:
    """Standard DCG with log2(i+1) discount, gain = relevance grade."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(ranked_grades: Sequence[int], k: int) -> float:
    """nDCG@k. ranked_grades is the sequence of grades for the top-k retrieved docs.

    Note: ideal DCG uses the same set of documents sorted by grade descending.
    For the standard nDCG over a per-query qrels set, callers should pass the
    ranked grades of all retrieved docs at top-k and the ideal grades from the
    full qrels sorted descending, padded if shorter than k.
    """
    if k <= 0:
        return 0.0
    actual = list(ranked_grades[:k])
    ideal = sorted(actual, reverse=True)
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(actual) / idcg


def ndcg_at_k_with_qrels(
    ranked_doc_ids: Sequence[str],
    qrels: dict[str, int],
    k: int,
) -> float:
    """nDCG@k where ideal ranking comes from the full qrels (not just top-k).

    qrels: doc_id -> grade for relevant docs (missing docs implicitly grade 0).
    """
    if k <= 0:
        return 0.0
    actual_grades = [qrels.get(did, 0) for did in ranked_doc_ids[:k]]
    ideal_grades = sorted(qrels.values(), reverse=True)[:k]
    idcg = dcg(ideal_grades)
    if idcg == 0:
        return 0.0
    return dcg(actual_grades) / idcg


def mrr(ranked_doc_ids: Sequence[str], qrels: dict[str, int]) -> float:
    """Mean reciprocal rank of the first relevant document (grade >= 1)."""
    for i, did in enumerate(ranked_doc_ids):
        if qrels.get(did, 0) >= 1:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_doc_ids: Sequence[str], qrels: dict[str, int], k: int) -> float:
    """Recall@k: fraction of relevant docs (grade >= 1) appearing in top-k."""
    relevant = {did for did, g in qrels.items() if g >= 1}
    if not relevant:
        return 0.0
    retrieved_top_k = set(ranked_doc_ids[:k])
    return len(relevant & retrieved_top_k) / len(relevant)


def mean_ndcg(per_query_ndcg: Sequence[float]) -> float:
    if not per_query_ndcg:
        return 0.0
    return sum(per_query_ndcg) / len(per_query_ndcg)
