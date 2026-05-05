"""Phase 1.2 + 2.2 driver: generate n=100 GTA scenarios using Ollama (qwen3.5)
and evaluate four gaze-based ranking methods on them.

Methods compared (matches paper §IV-D 4-way comparison):
    a) Random
    b) Current-Gaze-Only (use only the latest fixation target)
    c) Frequency-Only (rank by cumulative gaze frequency)
    d) GTA (the SCAR proposed method using directional + temporal patterns)

Reward: nDCG@5 over the (ground_truth + 4 distractors) set per scenario.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from src.gta.scenario_gen import (
    GTAScenario, OllamaBackend, TemplateBackend, generate_scenarios, scenarios_to_jsonable,
)
from src.utils.metrics import ndcg_at_k_with_qrels


def _docs_for_scenario(s: GTAScenario) -> list[str]:
    """Return all 5 candidate doc IDs in canonical order."""
    return [s.ground_truth_doc_id, *s.distractor_doc_ids]


def _rank_random(s: GTAScenario, rng: np.random.Generator) -> list[str]:
    docs = _docs_for_scenario(s)
    perm = rng.permutation(len(docs))
    return [docs[i] for i in perm]


def _rank_current_gaze_only(s: GTAScenario) -> list[str]:
    """Use only the LAST gaze target. Doc whose person matches goes first; rest by random tie."""
    if not s.gaze_trajectory:
        return _docs_for_scenario(s)
    last = s.gaze_trajectory[-1].target_person
    docs = _docs_for_scenario(s)
    def key(did: str) -> int:
        person = did.replace("d_target_", "").replace("d_distract_", "")
        return 0 if person == last else 1
    return sorted(docs, key=key)


def _rank_frequency_only(s: GTAScenario) -> list[str]:
    counts = Counter(e.target_person for e in s.gaze_trajectory)
    docs = _docs_for_scenario(s)
    def key(did: str) -> tuple[int, str]:
        person = did.replace("d_target_", "").replace("d_distract_", "")
        return (-counts.get(person, 0), did)  # higher count first; tie by id
    return sorted(docs, key=key)


def _rank_gta(s: GTAScenario) -> list[str]:
    """Trajectory-aware: weight by recency (later events more important) + duration."""
    weights: dict[str, float] = {}
    n = len(s.gaze_trajectory)
    for i, e in enumerate(s.gaze_trajectory):
        recency = (i + 1) / n  # newer events weighted higher
        weights[e.target_person] = weights.get(e.target_person, 0.0) + recency * (e.duration_ms / 1000.0)
    docs = _docs_for_scenario(s)
    def key(did: str) -> tuple[float, str]:
        person = did.replace("d_target_", "").replace("d_distract_", "")
        return (-weights.get(person, 0.0), did)
    return sorted(docs, key=key)


def _ndcg(ranked_doc_ids: list[str], gt: str) -> float:
    qrels = {gt: 2}
    return ndcg_at_k_with_qrels(ranked_doc_ids, qrels, k=5)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_ollama", action="store_true",
                   help="Use Ollama qwen3.5 to generate descriptions/queries (slower).")
    p.add_argument("--model", type=str, default="qwen3.5:latest")
    p.add_argument("--out", type=str, default="data/results/02_gta_n100.json")
    p.add_argument("--scenarios_out", type=str, default="data/corpora/gta_scenarios.json")
    args = p.parse_args()

    backend = OllamaBackend(model=args.model) if args.use_ollama else TemplateBackend()
    print(f"[gta] generating n={args.n} scenarios with backend={type(backend).__name__}")
    t0 = time.time()
    scenarios = generate_scenarios(n=args.n, backend=backend, seed=args.seed)
    gen_t = time.time() - t0
    print(f"[gta] generation took {gen_t:.2f}s")

    Path(args.scenarios_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.scenarios_out).write_text(
        json.dumps(scenarios_to_jsonable(scenarios), indent=2)
    )

    rng = np.random.default_rng(args.seed)
    method_scores: dict[str, list[float]] = {
        "random": [], "current_gaze_only": [], "frequency_only": [], "gta": [],
    }
    for s in scenarios:
        method_scores["random"].append(_ndcg(_rank_random(s, rng), s.ground_truth_doc_id))
        method_scores["current_gaze_only"].append(_ndcg(_rank_current_gaze_only(s), s.ground_truth_doc_id))
        method_scores["frequency_only"].append(_ndcg(_rank_frequency_only(s), s.ground_truth_doc_id))
        method_scores["gta"].append(_ndcg(_rank_gta(s), s.ground_truth_doc_id))

    summary = {}
    for k, v in method_scores.items():
        arr = np.array(v)
        summary[k] = {
            "mean_ndcg5": float(arr.mean()),
            "std_ndcg5": float(arr.std()),
            "n": int(len(arr)),
        }

    elapsed = time.time() - t0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": vars(args),
        "elapsed_seconds": elapsed,
        "n_scenarios": len(scenarios),
        "summary": summary,
    }, indent=2))

    print("\n[gta] method comparison (n={}):".format(len(scenarios)))
    for k, v in summary.items():
        print(f"  {k:<20} nDCG@5 = {v['mean_ndcg5']:.4f} +/- {v['std_ndcg5']:.4f}")
    print(f"\n[gta] wrote {out_path}")


if __name__ == "__main__":
    main()
