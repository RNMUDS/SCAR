"""Tests for src/baseline/linucb.py."""
import numpy as np

from src.baseline.linucb import (
    ARM_NAMES, CONTEXT_DIM, LinUCBArm, LinUCBPolicy,
    context_vector, make_linucb_base, make_linucb_warm, make_scar_linucb,
)
from src.utils.types import SpatialContext


def _ctx(role="engineer", zone="meeting_room") -> SpatialContext:
    return SpatialContext(
        user_id="u1", role=role, current_zone=zone, current_day=10,
        nearby_persons=(), gaze_target=None,
    )


def test_context_vector_shape_and_bias():
    x = context_vector(_ctx())
    assert x.shape == (CONTEXT_DIM,)
    assert x[-1] == 1.0  # bias always 1
    assert x.sum() == 3.0  # role + zone + bias


def test_context_vector_distinct_roles():
    x1 = context_vector(_ctx(role="engineer"))
    x2 = context_vector(_ctx(role="exec"))
    assert not np.allclose(x1, x2)


def test_linucb_arm_initial_identity():
    arm = LinUCBArm.fresh(d=5)
    assert np.allclose(arm.A, np.eye(5))
    assert np.allclose(arm.b, np.zeros(5))


def test_linucb_arm_update():
    arm = LinUCBArm.fresh(d=3)
    x = np.array([1.0, 0.0, 0.5])
    arm.update(x, reward=1.0)
    expected_A = np.eye(3) + np.outer(x, x)
    assert np.allclose(arm.A, expected_A)
    assert np.allclose(arm.b, x)


def test_linucb_arm_ucb_decreases_with_data():
    """Initially exploration bonus is high; after many same-context observations,
    pred dominates and exploration shrinks."""
    arm = LinUCBArm.fresh(d=3)
    x = np.array([1.0, 0.0, 0.0])
    _, expl_initial = arm.ucb(x, alpha=1.0)
    for _ in range(50):
        arm.update(x, reward=0.5)
    _, expl_later = arm.ucb(x, alpha=1.0)
    assert expl_later < expl_initial


def test_linucb_policy_picks_an_arm():
    policy = make_linucb_base()
    x = context_vector(_ctx())
    arm = policy.select_arm(x)
    assert arm in ARM_NAMES


def test_linucb_learns_a_signal():
    """If we always reward 'scar-full' = 1.0 and others = 0.0, the policy should
    converge to selecting 'scar-full' for any context."""
    rng = np.random.default_rng(0)
    policy = make_linucb_base(alpha=1.0, seed=0)
    n_warm = 200
    for _ in range(n_warm):
        ctx = _ctx(role=rng.choice(["engineer", "designer", "pm"]))
        x = context_vector(ctx)
        chosen = policy.select_arm(x)
        reward = 1.0 if chosen == "scar-full" else 0.0
        policy.update(chosen, x, reward)

    # Now check: given a fresh context, scar-full should be chosen with probability ~1.
    test_ctx = _ctx()
    x = context_vector(test_ctx)
    n_eval = 50
    picks = [policy.select_arm(x) for _ in range(n_eval)]
    scar_count = sum(1 for p in picks if p == "scar-full")
    assert scar_count / n_eval > 0.7, f"Expected >70% scar-full picks, got {scar_count}/{n_eval}"


def test_warm_start_biases_arm():
    """warm-started policy with scar-full prior should pick scar-full early."""
    policy = make_scar_linucb(alpha=0.5, seed=0, scar_full_boost=0.9, other_arm_prior=0.1)
    x = context_vector(_ctx())
    # Without any updates, the warm-started prior should make scar-full attractive.
    # We allow some exploration noise but expect scar-full pulled in majority.
    picks = [policy.select_arm(x) for _ in range(50)]
    scar_count = sum(1 for p in picks if p == "scar-full")
    assert scar_count >= 25, f"Expected >=25/50 scar-full picks from warm-start, got {scar_count}"


def test_three_variants_construct():
    """Smoke: all three named factories build valid policies."""
    for factory in (make_linucb_base, make_linucb_warm, make_scar_linucb):
        p = factory()
        assert isinstance(p, LinUCBPolicy)
        x = context_vector(_ctx())
        assert p.select_arm(x) in ARM_NAMES
