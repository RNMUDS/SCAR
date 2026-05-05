"""Tests for src/noise/inject.py."""
import random

from src.noise.inject import corrupt_context, corrupt_query, _flip_prob_from_sigma
from src.utils.types import SpatialContext, Query


def _ctx(**kw):
    base = dict(
        user_id="u1", role="engineer", current_zone="meeting_room",
        current_day=10, nearby_persons=("alice", "bob"), gaze_target="alice",
    )
    base.update(kw)
    return SpatialContext(**base)


def test_sigma_zero_unchanged():
    ctx = _ctx()
    rng = random.Random(0)
    out = corrupt_context(ctx, sigma=0.0, person_pool=["alice", "bob", "carol"], rng=rng)
    assert out == ctx


def test_flip_prob_clamping():
    assert _flip_prob_from_sigma(-1.0) == 0.0
    assert _flip_prob_from_sigma(0.0) == 0.0
    assert _flip_prob_from_sigma(0.3) == 0.3
    assert _flip_prob_from_sigma(2.0) == 0.5


def test_corrupt_context_changes_with_high_sigma():
    """With sigma=0.5 and 50 persons in pool, the measured nearby set should
    differ from the true set most of the time."""
    ctx = _ctx(nearby_persons=("alice", "bob"))
    pool = ["alice", "bob"] + [f"p{i}" for i in range(20)]
    rng = random.Random(123)
    diffs = 0
    for _ in range(20):
        out = corrupt_context(ctx, sigma=0.5, person_pool=pool, rng=rng)
        if set(out.nearby_persons) != set(ctx.nearby_persons):
            diffs += 1
    assert diffs >= 15, f"expected most trials to differ, got {diffs}/20"


def test_corrupt_query_returns_new_query():
    rng = random.Random(0)
    q = Query(query_id="q1", text="hello", context=_ctx())
    pool = ["alice", "bob", "carol"]
    out = corrupt_query(q, sigma=0.3, person_pool=pool, rng=rng)
    assert out.query_id == q.query_id  # immutable: new object, same id
    assert out.text == q.text


def test_corrupt_context_immutable():
    ctx = _ctx()
    rng = random.Random(0)
    pool = ["alice", "bob", "carol"]
    out = corrupt_context(ctx, sigma=0.4, person_pool=pool, rng=rng)
    # the original is unchanged
    assert ctx.nearby_persons == ("alice", "bob")
    # the output is a different object
    assert out is not ctx
