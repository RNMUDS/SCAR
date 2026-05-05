"""LinUCB contextual bandit for SCAR re-ranking strategy selection.

Adapted from PRIMA reference (06_reviewer_response/runReviewerExperiments3.js).

Bandit framing:
    - Arms: 6 re-ranking strategies (different SCAR weight vectors)
    - Context x: 11-dim (role one-hot 5 + zone one-hot 5 + bias 1)
    - Reward: nDCG@5 of the chosen strategy on this query (in [0, 1])
    - Update: per query (one observation per (arm, context, reward))

Three policy variants:
    1. linucb-base    : standard LinUCB
    2. linucb-warm    : warm-started by injecting synthetic obs from LLM/SCAR prior
    3. scar+linucb    : same as warm but the prior boosts the scar-full arm specifically
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from src.sim.scar_scorer import ScarWeights
from src.utils.types import ROLES, ZONE_TYPES, SpatialContext


# ---------------------------------------------------------------------------
# Arm definitions: each arm is a (name, ScarWeights) pair.
# Weights are normalized to sum to 1 (alpha + beta + gamma + delta + epsilon).
# ---------------------------------------------------------------------------

ARMS: tuple[tuple[str, ScarWeights], ...] = (
    ("bm25",        ScarWeights(alpha=1.0,  beta=0.0,  gamma=0.0,  delta=0.0,  epsilon=0.0)),
    ("bm25+prox",   ScarWeights(alpha=0.7,  beta=0.3,  gamma=0.0,  delta=0.0,  epsilon=0.0)),
    ("bm25+gaze",   ScarWeights(alpha=0.7,  beta=0.0,  gamma=0.3,  delta=0.0,  epsilon=0.0)),
    ("bm25+loc",    ScarWeights(alpha=0.7,  beta=0.0,  gamma=0.0,  delta=0.3,  epsilon=0.0)),
    ("bm25+time",   ScarWeights(alpha=0.7,  beta=0.0,  gamma=0.0,  delta=0.0,  epsilon=0.3)),
    ("scar-full",   ScarWeights(alpha=0.50, beta=0.20, gamma=0.15, delta=0.10, epsilon=0.05)),
)
ARM_NAMES: tuple[str, ...] = tuple(name for name, _ in ARMS)


# ---------------------------------------------------------------------------
# Context featurizer: 11D = role one-hot (5) + zone one-hot (5) + bias (1).
# ---------------------------------------------------------------------------

CONTEXT_DIM = len(ROLES) + len(ZONE_TYPES) + 1
_ROLE_INDEX = {r: i for i, r in enumerate(ROLES)}
_ZONE_INDEX = {z: i for i, z in enumerate(ZONE_TYPES)}


def context_vector(ctx: SpatialContext) -> np.ndarray:
    """Build the 11-dim context vector."""
    x = np.zeros(CONTEXT_DIM, dtype=np.float64)
    if ctx.role in _ROLE_INDEX:
        x[_ROLE_INDEX[ctx.role]] = 1.0
    if ctx.current_zone in _ZONE_INDEX:
        x[len(ROLES) + _ZONE_INDEX[ctx.current_zone]] = 1.0
    x[-1] = 1.0  # bias
    return x


# ---------------------------------------------------------------------------
# LinUCB core.
# ---------------------------------------------------------------------------

@dataclass
class LinUCBArm:
    """Per-arm sufficient statistics: A = I + sum(x x^T), b = sum(r * x)."""
    A: np.ndarray
    b: np.ndarray

    @classmethod
    def fresh(cls, d: int) -> "LinUCBArm":
        return cls(A=np.eye(d, dtype=np.float64), b=np.zeros(d, dtype=np.float64))

    def update(self, x: np.ndarray, reward: float) -> None:
        self.A += np.outer(x, x)
        self.b += reward * x

    def ucb(self, x: np.ndarray, alpha: float) -> tuple[float, float]:
        """Return (predicted_reward, exploration_bonus). UCB = pred + alpha * bonus."""
        Ainv = np.linalg.inv(self.A)
        theta = Ainv @ self.b
        pred = float(theta @ x)
        explore = float(np.sqrt(max(0.0, x @ Ainv @ x)))
        return pred, alpha * explore


@dataclass
class LinUCBPolicy:
    """LinUCB policy over a fixed set of arms.

    `seed`: RNG seed for tie-breaking only (the algorithm is deterministic
    otherwise, but UCB ties can occur at start when all arms are equivalent).
    """
    alpha: float = 1.0
    arm_names: Sequence[str] = field(default_factory=lambda: ARM_NAMES)
    context_dim: int = CONTEXT_DIM
    seed: int = 42
    _arms: dict[str, LinUCBArm] = field(default_factory=dict)
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        if not self._arms:
            self._arms = {name: LinUCBArm.fresh(self.context_dim) for name in self.arm_names}

    def select_arm(self, x: np.ndarray) -> str:
        """Return the arm name with the highest UCB. Ties broken by RNG."""
        best_score = -np.inf
        best_arms: list[str] = []
        for name in self.arm_names:
            pred, expl = self._arms[name].ucb(x, self.alpha)
            score = pred + expl
            if score > best_score + 1e-12:
                best_score = score
                best_arms = [name]
            elif score >= best_score - 1e-12:
                best_arms.append(name)
        if len(best_arms) == 1:
            return best_arms[0]
        return str(self._rng.choice(best_arms))

    def update(self, arm_name: str, x: np.ndarray, reward: float) -> None:
        self._arms[arm_name].update(x, reward)

    def warm_start(self, prior_obs: Sequence[tuple[str, np.ndarray, float]], n_replicas: int = 20) -> None:
        """Inject synthetic observations from a prior (LLM / SCAR defaults).

        prior_obs: list of (arm_name, x, expected_reward) tuples to replicate.
        n_replicas: how many copies of each prior obs to inject (PRIMA uses 20).
        """
        for arm_name, x, r in prior_obs:
            for _ in range(n_replicas):
                self._arms[arm_name].update(x, r)


# ---------------------------------------------------------------------------
# Three named variants matching system_design_v2.md.
# ---------------------------------------------------------------------------

def make_linucb_base(alpha: float = 1.0, seed: int = 42) -> LinUCBPolicy:
    return LinUCBPolicy(alpha=alpha, seed=seed)


def make_linucb_warm(
    alpha: float = 1.0,
    seed: int = 42,
    prior_reward_per_arm: Optional[dict[str, float]] = None,
    n_replicas: int = 20,
) -> LinUCBPolicy:
    """Warm-start with a uniform prior reward per arm. Useful when no SCAR prior is known.

    Default prior_reward_per_arm: each arm gets reward 0.5 (chance).
    """
    policy = LinUCBPolicy(alpha=alpha, seed=seed)
    if prior_reward_per_arm is None:
        prior_reward_per_arm = {name: 0.5 for name in ARM_NAMES}
    # Use a neutral context vector (all roles/zones equally likely).
    neutral = np.zeros(CONTEXT_DIM)
    neutral[-1] = 1.0
    prior_obs = [(name, neutral, r) for name, r in prior_reward_per_arm.items()]
    policy.warm_start(prior_obs, n_replicas=n_replicas)
    return policy


def make_scar_linucb(
    alpha: float = 1.0,
    seed: int = 42,
    scar_full_boost: float = 0.7,
    other_arm_prior: float = 0.45,
    n_replicas: int = 20,
) -> LinUCBPolicy:
    """SCAR-warmed LinUCB. The scar-full arm is given a higher prior reward,
    encoding the fact that SCAR with default weights is the paper's best fixed policy.
    """
    prior = {name: other_arm_prior for name in ARM_NAMES}
    prior["scar-full"] = scar_full_boost
    return make_linucb_warm(
        alpha=alpha,
        seed=seed,
        prior_reward_per_arm=prior,
        n_replicas=n_replicas,
    )
