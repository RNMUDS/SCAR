"""Evaluation harness: run a policy (LinUCB or fixed-arm) over a query stream.

Two modes:
    - `evaluate_fixed`: every query uses the same arm (e.g., scar-full or bm25).
    - `evaluate_bandit`: LinUCB selects an arm per query, gets reward, updates.

Returns per-query and aggregate metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.baseline.linucb import (
    ARM_NAMES,
    ARMS,
    LinUCBPolicy,
    context_vector,
)
from src.sim.scar_scorer import ScarWeights, score
from src.sim.text_sim import BM25Backend
from src.utils.metrics import ndcg_at_k_with_qrels, mrr, recall_at_k
from src.utils.types import Corpus, Query


@dataclass
class QueryResult:
    query_id: str
    arm: str
    ndcg5: float
    mrr: float
    recall5: float


@dataclass
class EvalReport:
    label: str
    per_query: tuple[QueryResult, ...]
    arm_pull_counts: dict[str, int]
    mean_ndcg5: float
    mean_mrr: float
    mean_recall5: float


def _rank_with_arm(
    query: Query,
    bm25: BM25Backend,
    documents,
    weights: ScarWeights,
) -> list[str]:
    bm25_scores = bm25.scores(query.text)
    scored = []
    for d in documents:
        ts = bm25_scores.get(d.doc_id, 0.0)
        r = score(ts, d, query.context, weights)
        scored.append(r)
    scored.sort(key=lambda r: r.final_score, reverse=True)
    return [r.doc_id for r in scored]


def _query_metrics(
    query: Query,
    ranked_doc_ids: list[str],
    qrels_per_query: dict[str, dict[str, int]],
    arm: str,
) -> QueryResult:
    qrels = qrels_per_query.get(query.query_id, {})
    return QueryResult(
        query_id=query.query_id,
        arm=arm,
        ndcg5=ndcg_at_k_with_qrels(ranked_doc_ids, qrels, k=5),
        mrr=mrr(ranked_doc_ids, qrels),
        recall5=recall_at_k(ranked_doc_ids, qrels, k=5),
    )


def evaluate_fixed_arm(
    corpus: Corpus,
    arm_name: str,
    bm25: Optional[BM25Backend] = None,
) -> EvalReport:
    bm25 = bm25 or BM25Backend.from_corpus(corpus)
    arm_weights = dict(ARMS)[arm_name]
    qrels_per_query = corpus.relevance_map()

    per_query: list[QueryResult] = []
    for q in corpus.queries:
        ranked = _rank_with_arm(q, bm25, corpus.documents, arm_weights)
        per_query.append(_query_metrics(q, ranked, qrels_per_query, arm_name))

    return _aggregate(per_query, label=arm_name)


def evaluate_bandit(
    corpus: Corpus,
    policy: LinUCBPolicy,
    bm25: Optional[BM25Backend] = None,
    label: str = "linucb",
) -> EvalReport:
    bm25 = bm25 or BM25Backend.from_corpus(corpus)
    arm_weights = dict(ARMS)
    qrels_per_query = corpus.relevance_map()

    per_query: list[QueryResult] = []
    for q in corpus.queries:
        x = context_vector(q.context)
        chosen_arm = policy.select_arm(x)
        ranked = _rank_with_arm(q, bm25, corpus.documents, arm_weights[chosen_arm])
        result = _query_metrics(q, ranked, qrels_per_query, chosen_arm)
        # reward = nDCG@5 (in [0, 1]) — primary signal for the bandit
        policy.update(chosen_arm, x, result.ndcg5)
        per_query.append(result)

    return _aggregate(per_query, label=label)


def _aggregate(per_query: list[QueryResult], label: str) -> EvalReport:
    if not per_query:
        return EvalReport(
            label=label, per_query=(), arm_pull_counts={},
            mean_ndcg5=0.0, mean_mrr=0.0, mean_recall5=0.0,
        )
    counts: dict[str, int] = {}
    for r in per_query:
        counts[r.arm] = counts.get(r.arm, 0) + 1
    return EvalReport(
        label=label,
        per_query=tuple(per_query),
        arm_pull_counts=counts,
        mean_ndcg5=float(np.mean([r.ndcg5 for r in per_query])),
        mean_mrr=float(np.mean([r.mrr for r in per_query])),
        mean_recall5=float(np.mean([r.recall5 for r in per_query])),
    )
