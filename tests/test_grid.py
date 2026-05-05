"""Tests for src/grid/lhs_search.py."""
import numpy as np

from src.grid.lhs_search import lhs_simplex_5d, evaluate_grid, best_point
from src.sim.scar_scorer import DEFAULT_WEIGHTS, ScarWeights


def test_lhs_size():
    pts = lhs_simplex_5d(n_points=81, seed=42)
    assert len(pts) == 81


def test_lhs_first_is_default():
    pts = lhs_simplex_5d(n_points=81, seed=42)
    assert pts[0].as_array() == DEFAULT_WEIGHTS


def test_lhs_simplex_constraint():
    pts = lhs_simplex_5d(n_points=50, seed=42)
    for p in pts:
        s = sum(p.as_array())
        assert abs(s - 1.0) < 1e-9, f"sum != 1.0: {s} for {p}"
        assert all(w >= -1e-12 for w in p.as_array())


def test_lhs_diversity():
    """At least 30 of 81 points should differ from each other coordinate-wise (>=0.05)."""
    pts = lhs_simplex_5d(n_points=81, seed=42)
    arr = np.array([p.as_array() for p in pts])
    # std across the 81 points in each coord should be > 0.05
    stds = arr.std(axis=0)
    assert all(s > 0.05 for s in stds), f"low diversity: {stds}"


def test_evaluate_grid_runs():
    pts = lhs_simplex_5d(n_points=10, seed=42)
    scored = evaluate_grid(pts, eval_fn=lambda w: w.alpha)
    assert len(scored) == 10
    best, best_score = best_point(scored)
    # the point with highest alpha wins under alpha-only objective
    assert best_score == max(p.alpha for p in pts)
