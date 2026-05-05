"""Latin Hypercube Sampling on the 5-simplex for SCAR weight tuning.

The weight space is constrained: alpha + beta + gamma + delta + epsilon = 1,
each w_i >= 0. We sample 81 points roughly uniformly over this simplex.

Method: sample 81 points from a 4-dim LHS in [0, 1]^4, then project into a
sorted 4-tuple (s_1, s_2, s_3, s_4) and form simplex coordinates as
(s_1, s_2 - s_1, s_3 - s_2, s_4 - s_3, 1 - s_4). This is the standard
"sorted uniform" trick for uniform Dirichlet on the simplex.

We always include the paper default weights as point 0 so we can compare
the LHS sample against the paper's chosen point.
"""
from __future__ import annotations

import numpy as np

from src.sim.scar_scorer import DEFAULT_WEIGHTS, ScarWeights


def _lhs_unit(n_points: int, n_dim: int, seed: int) -> np.ndarray:
    """Standard Latin Hypercube Sampling in [0, 1]^n_dim."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_points, n_dim), dtype=np.float64)
    for j in range(n_dim):
        perm = rng.permutation(n_points)
        jitter = rng.uniform(size=n_points)
        out[:, j] = (perm + jitter) / n_points
    return out


def lhs_simplex_5d(n_points: int = 81, seed: int = 42) -> list[ScarWeights]:
    """Generate n_points weight tuples on the 5-simplex via LHS.

    Always includes the paper default weights as the first point.
    """
    raw = _lhs_unit(n_points - 1, n_dim=4, seed=seed)
    raw_sorted = np.sort(raw, axis=1)
    # Differences yield 5 simplex coordinates
    diffs = np.zeros((n_points - 1, 5))
    diffs[:, 0] = raw_sorted[:, 0]
    diffs[:, 1] = raw_sorted[:, 1] - raw_sorted[:, 0]
    diffs[:, 2] = raw_sorted[:, 2] - raw_sorted[:, 1]
    diffs[:, 3] = raw_sorted[:, 3] - raw_sorted[:, 2]
    diffs[:, 4] = 1.0 - raw_sorted[:, 3]

    # Permute the assignment of simplex coordinates to (alpha, beta, gamma, delta, epsilon)
    # to avoid clustering on the alpha axis. Use a deterministic Sobol-like reorder.
    rng = np.random.default_rng(seed + 1)
    perms = rng.permuted(
        np.tile(np.arange(5), (n_points - 1, 1)), axis=1,
    )
    permuted = np.take_along_axis(diffs, perms, axis=1)

    points = [ScarWeights(*DEFAULT_WEIGHTS)]
    for row in permuted:
        a, b, c, d, e = (float(x) for x in row)
        points.append(ScarWeights(alpha=a, beta=b, gamma=c, delta=d, epsilon=e))
    return points


def evaluate_grid(
    points: list[ScarWeights],
    eval_fn,
) -> list[tuple[ScarWeights, float]]:
    """Apply eval_fn(weights) -> mean nDCG@5 to every point. Returns list of (weights, score)."""
    return [(w, eval_fn(w)) for w in points]


def best_point(scored: list[tuple[ScarWeights, float]]) -> tuple[ScarWeights, float]:
    return max(scored, key=lambda kv: kv[1])
