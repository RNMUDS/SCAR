"""Gaussian noise injection on spatial signals.

Two injectors:
    - proximity_noise: corrupts the SpatialContext.nearby_persons membership
      (each true neighbor flipped to non-neighbor with probability p_flip
      determined by sigma; each true non-neighbor flipped to neighbor likewise)
    - gaze_noise: with probability p, replaces gaze_target with a random
      person from nearby_persons (or None) — simulates gaze tracker drift.

We model noise as observation corruption (the *measured* spatial state differs
from the *true* state). The corrupted context is fed to SCAR; ground-truth
relevance still uses the true state.
"""
from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Optional, Sequence

from src.utils.types import Query, SpatialContext


def _flip_prob_from_sigma(sigma: float) -> float:
    """Map sigma in [0, 0.5] roughly to a flip probability in [0, 0.5]."""
    return min(0.5, max(0.0, sigma))


def corrupt_context(
    ctx: SpatialContext,
    sigma: float,
    person_pool: Sequence[str],
    rng: random.Random,
) -> SpatialContext:
    """Return a new SpatialContext with proximity and gaze corrupted.

    sigma=0 => unchanged. sigma=0.5 => maximum noise.
    """
    if sigma <= 0.0:
        return ctx

    p_flip = _flip_prob_from_sigma(sigma)
    true_nearby = set(ctx.nearby_persons)
    candidates = [p for p in person_pool if p != ctx.user_id]
    measured: list[str] = []
    for p in candidates:
        in_true = p in true_nearby
        if rng.random() < p_flip:
            # flipped: include if not in true, exclude if in true
            if not in_true:
                measured.append(p)
        else:
            # unflipped: keep as-is
            if in_true:
                measured.append(p)

    # Gaze: with probability p_flip, swap to random member of measured (or None)
    if ctx.gaze_target is not None and rng.random() < p_flip:
        if measured:
            new_gaze: Optional[str] = rng.choice(measured)
        else:
            new_gaze = None
    else:
        new_gaze = ctx.gaze_target
        # If old gaze target was dropped from measured nearby, set to None
        if new_gaze is not None and new_gaze not in measured and new_gaze not in true_nearby:
            new_gaze = None

    return replace(ctx, nearby_persons=tuple(measured), gaze_target=new_gaze)


def corrupt_query(
    query: Query,
    sigma: float,
    person_pool: Sequence[str],
    rng: random.Random,
) -> Query:
    return replace(query, context=corrupt_context(query.context, sigma, person_pool, rng))
