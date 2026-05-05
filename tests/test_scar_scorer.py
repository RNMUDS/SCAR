"""Tests for src/sim/scar_scorer.py."""
import math

from src.sim.scar_scorer import (
    DEFAULT_WEIGHTS, ScarWeights, prox_bonus, gaze_bonus, loc_bonus,
    time_decay, score, feature_vector,
)
from src.utils.types import Document, SpatialContext


def _doc(**kw) -> Document:
    base = dict(
        doc_id="d1", text="hello", author="alice", location_type="meeting_room",
        kind="meeting_log", created_day=10, mentions=(),
    )
    base.update(kw)
    return Document(**base)


def _ctx(**kw) -> SpatialContext:
    base = dict(
        user_id="u1", role="engineer", current_zone="meeting_room",
        current_day=15, nearby_persons=(), gaze_target=None,
    )
    base.update(kw)
    return SpatialContext(**base)


# ---- proximity ----

def test_prox_bonus_author_nearby():
    assert prox_bonus(_doc(author="alice"), _ctx(nearby_persons=("alice",))) == 1.0


def test_prox_bonus_mention_nearby():
    d = _doc(author="bob", mentions=("alice",))
    assert prox_bonus(d, _ctx(nearby_persons=("alice",))) == 0.5


def test_prox_bonus_none():
    assert prox_bonus(_doc(author="bob"), _ctx(nearby_persons=("carol",))) == 0.0


# ---- gaze ----

def test_gaze_bonus_author_match():
    assert gaze_bonus(_doc(author="alice"), _ctx(gaze_target="alice")) == 1.0


def test_gaze_bonus_mention_match():
    assert gaze_bonus(_doc(author="bob", mentions=("alice",)), _ctx(gaze_target="alice")) == 0.6


def test_gaze_bonus_no_target():
    assert gaze_bonus(_doc(author="alice"), _ctx(gaze_target=None)) == 0.0


# ---- location ----

def test_loc_bonus_zone_match():
    assert loc_bonus(_doc(location_type="meeting_room"), _ctx(current_zone="meeting_room")) == 1.0


def test_loc_bonus_zone_affinity():
    # current zone meeting_room, doc kind meeting_log but in different location.
    d = _doc(location_type="library", kind="meeting_log")
    assert loc_bonus(d, _ctx(current_zone="meeting_room")) == 0.7


def test_loc_bonus_no_match():
    d = _doc(location_type="library", kind="other")
    assert loc_bonus(d, _ctx(current_zone="meeting_room")) == 0.0


# ---- time ----

def test_time_decay_zero_days():
    assert abs(time_decay(_doc(created_day=15), _ctx(current_day=15)) - 1.0) < 1e-9


def test_time_decay_seven_days():
    # half-life: 7 days -> 0.5
    assert abs(time_decay(_doc(created_day=8), _ctx(current_day=15)) - 0.5) < 1e-9


def test_time_decay_negative_clipped():
    # future doc — clipped to 0 days delta
    assert time_decay(_doc(created_day=20), _ctx(current_day=15)) == 1.0


# ---- composite score ----

def test_score_default_weights_sum():
    a, b, c, d, e = DEFAULT_WEIGHTS
    assert abs(a + b + c + d + e - 1.0) < 1e-9


def test_score_decomposition():
    # All bonuses 1.0, text_sim 1.0 -> final score = sum of weights = 1.0
    doc = _doc(author="alice", location_type="meeting_room", created_day=15, mentions=())
    ctx = _ctx(current_zone="meeting_room", nearby_persons=("alice",), gaze_target="alice")
    r = score(text_sim=1.0, doc=doc, ctx=ctx, weights=ScarWeights())
    assert abs(r.final_score - sum(DEFAULT_WEIGHTS)) < 1e-9
    assert r.prox_bonus == 1.0
    assert r.gaze_bonus == 1.0
    assert r.loc_bonus == 1.0
    assert abs(r.time_decay - 1.0) < 1e-9


def test_feature_vector_shape():
    fv = feature_vector(_doc(), _ctx())
    assert len(fv) == 4
    assert all(isinstance(v, float) for v in fv)
