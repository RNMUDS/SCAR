"""Tests for src/utils/metrics.py."""
import math

from src.utils.metrics import (
    dcg, ndcg_at_k, ndcg_at_k_with_qrels, mrr, recall_at_k
)


def test_dcg_zero_for_all_zero():
    assert dcg([0, 0, 0]) == 0.0


def test_dcg_known_values():
    # DCG([3, 2, 3]) = 3/log2(2) + 2/log2(3) + 3/log2(4)
    # = 3/1 + 2/1.5849... + 3/2 = 3 + 1.2618 + 1.5 = 5.7618
    val = dcg([3, 2, 3])
    expected = 3.0 + 2.0 / math.log2(3) + 3.0 / math.log2(4)
    assert abs(val - expected) < 1e-9


def test_ndcg_perfect_ranking_is_1():
    assert ndcg_at_k([3, 2, 1], k=3) == 1.0


def test_ndcg_at_k_with_qrels_perfect():
    qrels = {"a": 2, "b": 1, "c": 0}
    # ranked perfectly
    assert ndcg_at_k_with_qrels(["a", "b", "c"], qrels, k=3) == 1.0


def test_ndcg_at_k_with_qrels_worst():
    qrels = {"a": 2, "b": 1, "c": 0}
    val = ndcg_at_k_with_qrels(["c", "b", "a"], qrels, k=3)
    assert 0 < val < 1


def test_mrr_first_position():
    assert mrr(["a", "b"], {"a": 2}) == 1.0


def test_mrr_second_position():
    assert mrr(["a", "b"], {"b": 1}) == 0.5


def test_mrr_no_relevant():
    assert mrr(["a", "b"], {}) == 0.0


def test_recall_at_k():
    qrels = {"a": 1, "b": 1, "c": 1, "d": 0}
    # 2 of 3 relevant in top-3
    assert recall_at_k(["a", "b", "x"], qrels, k=3) == 2 / 3
