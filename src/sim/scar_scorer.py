"""SCAR scoring function. Implements equations (Eq. 5)-(Eq. 10) from paper.tex Section V.

S(q,d,C) = alpha * TextSim(q,d)
         + beta  * ProxBonus(d,C)
         + gamma * GazeBonus(d,C)
         + delta * LocBonus(d,C)
         + epsilon * TimeDecay(d,C)

Default weights: alpha=0.50, beta=0.20, gamma=0.15, delta=0.10, epsilon=0.05.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from src.utils.types import Document, RetrievalResult, SpatialContext


DEFAULT_WEIGHTS = (0.50, 0.20, 0.15, 0.10, 0.05)
TIME_DECAY_LAMBDA = math.log(2.0) / 7.0  # 7-day half-life


# Zone-specific document-kind affinities for LocationBonus partial credit.
# When a doc's authoring zone differs from the user's current zone but the
# kind matches a zone-typical activity, we award a partial 0.7 bonus.
ZONE_AFFINITY = {
    "meeting_room": "meeting_log",
    "focus_booth": "spec_doc",
    "lounge": "brainstorm_note",
}


@dataclass(frozen=True)
class ScarWeights:
    alpha: float = DEFAULT_WEIGHTS[0]
    beta: float = DEFAULT_WEIGHTS[1]
    gamma: float = DEFAULT_WEIGHTS[2]
    delta: float = DEFAULT_WEIGHTS[3]
    epsilon: float = DEFAULT_WEIGHTS[4]

    def as_array(self) -> tuple[float, float, float, float, float]:
        return (self.alpha, self.beta, self.gamma, self.delta, self.epsilon)


def prox_bonus(doc: Document, ctx: SpatialContext) -> float:
    """Eq. 7: 1.0 if author is nearby, 0.5 if doc mentions a nearby person."""
    nearby = set(ctx.nearby_persons)
    if doc.author in nearby:
        return 1.0
    if any(p in nearby for p in doc.mentions):
        return 0.5
    return 0.0


def gaze_bonus(doc: Document, ctx: SpatialContext) -> float:
    """Eq. 8: 1.0 if author == gaze target, 0.6 if mentions gaze target."""
    target = ctx.gaze_target
    if target is None:
        return 0.0
    if doc.author == target:
        return 1.0
    if target in doc.mentions:
        return 0.6
    return 0.0


def loc_bonus(doc: Document, ctx: SpatialContext) -> float:
    """Eq. 9: 1.0 if doc location matches current zone; 0.7 for zone-affinity match."""
    if doc.location_type == ctx.current_zone:
        return 1.0
    affinity_kind = ZONE_AFFINITY.get(ctx.current_zone)
    if affinity_kind is not None and doc.kind == affinity_kind:
        return 0.7
    return 0.0


def time_decay(doc: Document, ctx: SpatialContext) -> float:
    """Eq. 10: exp(-lambda * delta_days), lambda = ln(2)/7. Negative delta clipped to 0."""
    delta_days = max(0, ctx.current_day - doc.created_day)
    return math.exp(-TIME_DECAY_LAMBDA * delta_days)


def score(
    text_sim: float,
    doc: Document,
    ctx: SpatialContext,
    weights: ScarWeights = ScarWeights(),
) -> RetrievalResult:
    """Compute the composite SCAR score for one (query, doc, context) tuple.

    Caller supplies text_sim (cosine sim or BM25 normalized).
    """
    p = prox_bonus(doc, ctx)
    g = gaze_bonus(doc, ctx)
    l = loc_bonus(doc, ctx)
    t = time_decay(doc, ctx)
    s = (
        weights.alpha * text_sim
        + weights.beta * p
        + weights.gamma * g
        + weights.delta * l
        + weights.epsilon * t
    )
    return RetrievalResult(
        doc_id=doc.doc_id,
        final_score=s,
        text_sim=text_sim,
        prox_bonus=p,
        gaze_bonus=g,
        loc_bonus=l,
        time_decay=t,
    )


def rank(
    candidates: Iterable[tuple[Document, float]],
    ctx: SpatialContext,
    weights: ScarWeights = ScarWeights(),
) -> list[RetrievalResult]:
    """Re-rank a candidate set by composite SCAR score (descending)."""
    results = [score(text_sim, doc, ctx, weights) for doc, text_sim in candidates]
    return sorted(results, key=lambda r: r.final_score, reverse=True)


def feature_vector(doc: Document, ctx: SpatialContext) -> tuple[float, float, float, float]:
    """Spatial-only feature vector x = [prox, gaze, loc, time]. Used by LinUCB."""
    return (
        prox_bonus(doc, ctx),
        gaze_bonus(doc, ctx),
        loc_bonus(doc, ctx),
        time_decay(doc, ctx),
    )
