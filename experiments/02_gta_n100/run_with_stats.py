"""GTA n=100 with multi-seed variance + paired stats.

For each (method_a, method_b) pair, computes paired t-test p-value and Cohen's d
across queries (per-query nDCG@5 differences). Applies Holm-Bonferroni correction.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from src.gta.scenario_gen import GTAScenario, TemplateBackend, generate_scenarios
from src.utils.metrics import ndcg_at_k_with_qrels
from src.utils.stats import cohens_d_paired, holm_bonferroni, paired_ttest_p


def _docs_for_scenario(s):
    return [s.ground_truth_doc_id, *s.distractor_doc_ids]


def _rank_random(s, rng):
    docs = _docs_for_scenario(s)
    perm = rng.permutation(len(docs))
    return [docs[i] for i in perm]


def _rank_current_gaze_only(s):
    if not s.gaze_trajectory:
        return _docs_for_scenario(s)
    last = s.gaze_trajectory[-1].target_person
    docs = _docs_for_scenario(s)
    return sorted(docs, key=lambda did: 0 if did.replace("d_target_", "").replace("d_distract_", "") == last else 1)


def _rank_frequency_only(s):
    counts = Counter(e.target_person for e in s.gaze_trajectory)
    docs = _docs_for_scenario(s)
    return sorted(docs, key=lambda did: (-counts.get(did.replace("d_target_", "").replace("d_distract_", ""), 0), did))


def _rank_gta(s):
    weights = {}
    n = len(s.gaze_trajectory)
    for i, e in enumerate(s.gaze_trajectory):
        recency = (i + 1) / n
        weights[e.target_person] = weights.get(e.target_person, 0.0) + recency * (e.duration_ms / 1000.0)
    docs = _docs_for_scenario(s)
    return sorted(docs, key=lambda did: (-weights.get(did.replace("d_target_", "").replace("d_distract_", ""), 0), did))


def _ndcg(ranked, gt):
    return ndcg_at_k_with_qrels(ranked, {gt: 2}, k=5)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seeds", type=str, default="42,137,271,314,1729")
    p.add_argument("--out", type=str, default="data/results/02_gta_with_stats.json")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    methods = ["random", "current_gaze_only", "frequency_only", "gta"]

    # per_seed_per_method[seed][method] = list of per-scenario nDCG@5 values
    per_seed_per_method: dict[int, dict[str, list[float]]] = {}
    t0 = time.time()
    for seed in seeds:
        scenarios = generate_scenarios(n=args.n, backend=TemplateBackend(), seed=seed)
        rng = np.random.default_rng(seed)
        per_seed_per_method[seed] = {m: [] for m in methods}
        for s in scenarios:
            per_seed_per_method[seed]["random"].append(_ndcg(_rank_random(s, rng), s.ground_truth_doc_id))
            per_seed_per_method[seed]["current_gaze_only"].append(_ndcg(_rank_current_gaze_only(s), s.ground_truth_doc_id))
            per_seed_per_method[seed]["frequency_only"].append(_ndcg(_rank_frequency_only(s), s.ground_truth_doc_id))
            per_seed_per_method[seed]["gta"].append(_ndcg(_rank_gta(s), s.ground_truth_doc_id))

    # Aggregate per method across all seeds (concatenate paired observations)
    pooled: dict[str, list[float]] = {m: [] for m in methods}
    for seed in seeds:
        for m in methods:
            pooled[m].extend(per_seed_per_method[seed][m])

    # Pairwise paired-t and Cohen's d, then Holm-Bonferroni correction
    pairs = [
        ("gta", "random"),
        ("gta", "current_gaze_only"),
        ("gta", "frequency_only"),
        ("current_gaze_only", "random"),
        ("frequency_only", "random"),
        ("current_gaze_only", "frequency_only"),
    ]
    raw_p = []
    pair_stats = []
    for a, b in pairs:
        p_val = paired_ttest_p(pooled[a], pooled[b])
        d = cohens_d_paired(pooled[a], pooled[b])
        raw_p.append(p_val)
        pair_stats.append({"a": a, "b": b, "p": p_val, "cohens_d": d})

    rejected = holm_bonferroni(raw_p, alpha=0.05)
    for entry, rej in zip(pair_stats, rejected):
        entry["holm_reject_at_005"] = rej

    summary = {}
    for m in methods:
        arr = np.array(pooled[m])
        summary[m] = {
            "mean_ndcg5": float(arr.mean()),
            "std_ndcg5": float(arr.std()),
            "n_obs": int(len(arr)),
        }

    elapsed = time.time() - t0
    print(f"\n[gta-stats] pooled n_obs per method: {len(pooled['gta'])}")
    for m, v in summary.items():
        print(f"  {m:<22} {v['mean_ndcg5']:.4f} +/- {v['std_ndcg5']:.4f}")
    print()
    print(f"  {'pair':<35} {'p':<10} {'d':<8} {'Holm':<6}")
    for s in pair_stats:
        print(f"  {s['a']:<14} vs {s['b']:<18} {s['p']:.4g}    {s['cohens_d']:+.3f}   {'YES' if s['holm_reject_at_005'] else 'no'}")
    print(f"\n[gta-stats] elapsed {elapsed:.2f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": vars(args),
        "n_seeds": len(seeds),
        "n_per_seed": args.n,
        "summary": summary,
        "pair_stats": pair_stats,
        "elapsed_seconds": elapsed,
    }, indent=2))
    print(f"[gta-stats] wrote {out_path}")


if __name__ == "__main__":
    main()
