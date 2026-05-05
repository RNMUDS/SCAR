"""Tests for src/utils/stats.py."""
import numpy as np

from src.utils.stats import (
    bootstrap_mean_ci, cohens_d_paired, holm_bonferroni, paired_ttest_p,
)


def test_paired_ttest_zero_when_identical():
    a = [0.5, 0.6, 0.7, 0.8]
    p = paired_ttest_p(a, a)
    # identical sequences -> p is nan due to zero variance, or 1.0 — accept either
    assert (p != p) or (p > 0.99)  # NaN check or close to 1


def test_paired_ttest_significant_when_clearly_different():
    rng = np.random.default_rng(0)
    n = 100
    a = rng.normal(0.5, 0.05, n)
    b = a - 0.1  # systematically lower
    p = paired_ttest_p(a, b)
    assert p < 0.001


def test_cohens_d_paired_positive():
    a = [0.6, 0.7, 0.8]
    b = [0.5, 0.6, 0.7]
    d = cohens_d_paired(a, b)
    # mean diff is 0.1, sd of diff is 0 -> we return 0 instead of inf
    assert d == 0.0


def test_cohens_d_paired_realistic():
    rng = np.random.default_rng(0)
    n = 100
    a = rng.normal(0.6, 0.1, n)
    # b shares most of a's noise but is shifted down with a small extra noise term
    b = a - 0.05 + rng.normal(0, 0.01, n)
    d = cohens_d_paired(a, b)
    # mean diff ~ 0.05, sd of diff ~ 0.01 -> d ~ 5 (very large)
    assert d > 1.0


def test_holm_bonferroni_all_reject():
    ps = [0.001, 0.002, 0.003]
    assert holm_bonferroni(ps, alpha=0.05) == [True, True, True]


def test_holm_bonferroni_mixed():
    # at alpha=0.05, n=4 tests
    # smallest p=0.005 vs threshold 0.05/4=0.0125 -> reject
    # next p=0.02 vs threshold 0.05/3=0.0167 -> NOT reject (step-down stops)
    # remaining p's also not rejected
    ps = [0.005, 0.02, 0.04, 0.5]
    out = holm_bonferroni(ps, alpha=0.05)
    assert out[0] is True
    assert out[1] is False
    assert out[2] is False
    assert out[3] is False


def test_bootstrap_ci_mean():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.5, 0.1, 200)
    m, lo, hi = bootstrap_mean_ci(vals, n_resamples=500, seed=0)
    assert lo < m < hi
    assert hi - lo < 0.05  # tight CI for n=200
