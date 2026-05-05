"""Statistical utilities for paired comparisons across queries.

Provides:
    - paired_ttest_p: paired t-test p-value (two-sided)
    - cohens_d_paired: paired Cohen's d (within-subject effect size)
    - holm_bonferroni: multiple-comparison correction
    - bootstrap_ci: 95% bootstrap CI on the mean
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


def paired_ttest_p(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided paired t-test p-value for H0: mean(a - b) = 0."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if len(a_arr) != len(b_arr):
        raise ValueError("paired_ttest requires equal-length sequences")
    if len(a_arr) < 2:
        return float("nan")
    res = stats.ttest_rel(a_arr, b_arr)
    return float(res.pvalue)


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired (within-subject) Cohen's d. d = mean(a-b) / sd(a-b)."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    diff = a_arr - b_arr
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    if sd < 1e-12:
        return 0.0
    return float(np.mean(diff) / sd)


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down correction. Returns reject/no-reject per test."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    rejected = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (n - rank)
        if p <= threshold:
            rejected[orig_idx] = True
        else:
            # once we fail to reject, all later (larger) p-values also fail
            break
    return rejected


def bootstrap_mean_ci(
    values: Sequence[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap (mean, lower CI, upper CI) at the given confidence level."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = []
    n = len(arr)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means.append(float(arr[idx].mean()))
    arr_means = np.asarray(means)
    lo_q = (1 - confidence) / 2 * 100
    hi_q = (1 + confidence) / 2 * 100
    return (
        float(arr.mean()),
        float(np.percentile(arr_means, lo_q)),
        float(np.percentile(arr_means, hi_q)),
    )
