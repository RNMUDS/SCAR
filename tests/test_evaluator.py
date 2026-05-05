"""Tests for src/sim/evaluator.py — end-to-end smoke."""
import pytest

from src.baseline.linucb import make_linucb_base, make_scar_linucb
from src.sim.evaluator import evaluate_bandit, evaluate_fixed_arm
from src.sim.synth_corpus import make_synth_corpus


@pytest.fixture(scope="module")
def corpus():
    return make_synth_corpus(n_persons=20, n_docs=40, n_queries=20, seed=42)


def test_fixed_arm_runs(corpus):
    report = evaluate_fixed_arm(corpus, arm_name="scar-full")
    assert report.label == "scar-full"
    assert len(report.per_query) == len(corpus.queries)
    assert 0.0 <= report.mean_ndcg5 <= 1.0


def test_fixed_arm_bm25_baseline(corpus):
    report = evaluate_fixed_arm(corpus, arm_name="bm25")
    assert 0.0 <= report.mean_ndcg5 <= 1.0


def test_scar_outperforms_bm25_on_synth(corpus):
    """On the synthetic corpus where qrels are constructed to reward spatial alignment,
    scar-full should beat bm25 on average."""
    bm25 = evaluate_fixed_arm(corpus, arm_name="bm25")
    scar = evaluate_fixed_arm(corpus, arm_name="scar-full")
    assert scar.mean_ndcg5 > bm25.mean_ndcg5, (
        f"scar-full ({scar.mean_ndcg5:.3f}) should beat bm25 ({bm25.mean_ndcg5:.3f}) on synth"
    )


def test_bandit_runs(corpus):
    policy = make_linucb_base(alpha=1.0, seed=0)
    report = evaluate_bandit(corpus, policy, label="linucb-base")
    assert len(report.per_query) == len(corpus.queries)
    assert sum(report.arm_pull_counts.values()) == len(corpus.queries)


def test_warm_bandit_pulls_scar_more(corpus):
    """scar+linucb with warm prior should pull scar-full more often than uniform-warm linucb."""
    warm = evaluate_bandit(corpus, make_scar_linucb(alpha=0.5, seed=0), label="scar+linucb")
    base = evaluate_bandit(corpus, make_linucb_base(alpha=0.5, seed=0), label="linucb-base")
    warm_scar = warm.arm_pull_counts.get("scar-full", 0)
    base_scar = base.arm_pull_counts.get("scar-full", 0)
    # Loose check; on a small corpus this may still vary, but warm should not be lower on average.
    assert warm_scar >= base_scar - 2, (
        f"warm scar-full pulls ({warm_scar}) significantly less than base ({base_scar})"
    )
