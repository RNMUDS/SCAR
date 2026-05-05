"""Phase 2 master driver: multi-seed re-experiments with paired stats and Holm.

Runs the headline comparisons on:
    - synthetic corpus (n=120 queries, 5 seeds)
    - GitHub Discussions real corpus (vercel/next.js, 1 seed because the corpus is fixed)

Pairs computed:
    - SCAR vs BM25
    - SCAR vs each single-feature variant
    - SCAR vs LinUCB-base
    - SCAR vs scar+linucb (warm-started)

Output: data/results/phase2_master.json — feeds figures and paper tables.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.baseline.linucb import (
    ARM_NAMES, make_linucb_base, make_linucb_warm, make_scar_linucb,
)
from src.corpus.github_ingest import discussions_to_corpus, fetch_answered_discussions
from src.sim.evaluator import evaluate_bandit, evaluate_fixed_arm
from src.sim.synth_corpus import make_synth_corpus
from src.sim.text_sim import BM25Backend
from src.utils.stats import bootstrap_mean_ci, cohens_d_paired, holm_bonferroni, paired_ttest_p
from src.utils.types import Corpus


def _per_query_ndcg(report) -> list[float]:
    return [r.ndcg5 for r in report.per_query]


def _evaluate_all_methods(corpus: Corpus, seed: int, alpha: float = 1.0) -> dict[str, list[float]]:
    bm25 = BM25Backend.from_corpus(corpus)
    out: dict[str, list[float]] = {}
    for arm in ARM_NAMES:
        out[arm] = _per_query_ndcg(evaluate_fixed_arm(corpus, arm, bm25=bm25))
    for label, factory in [
        ("linucb-base", lambda: make_linucb_base(alpha=alpha, seed=seed)),
        ("linucb-warm", lambda: make_linucb_warm(alpha=alpha, seed=seed)),
        ("scar+linucb", lambda: make_scar_linucb(alpha=alpha, seed=seed)),
    ]:
        out[label] = _per_query_ndcg(evaluate_bandit(corpus, factory(), bm25=bm25, label=label))
    return out


def _compute_pair_stats(method_to_per_query: dict[str, list[float]],
                        primary: str = "scar-full") -> list[dict]:
    """Pairwise tests vs primary method."""
    others = [m for m in method_to_per_query if m != primary]
    raw_p = []
    rows = []
    for o in others:
        a = method_to_per_query[primary]
        b = method_to_per_query[o]
        p = paired_ttest_p(a, b)
        d = cohens_d_paired(a, b)
        raw_p.append(p)
        rows.append({"a": primary, "b": o, "p": p, "cohens_d": d, "delta_mean": float(np.mean(a) - np.mean(b))})
    rejected = holm_bonferroni(raw_p, alpha=0.05)
    for r, rj in zip(rows, rejected):
        r["holm_reject_at_005"] = rj
    return rows


def _summary_for_corpus(name: str, method_to_per_query: dict[str, list[float]]) -> dict:
    summary = {}
    for m, vals in method_to_per_query.items():
        arr = np.asarray(vals)
        mean, lo, hi = bootstrap_mean_ci(arr, n_resamples=1000, seed=0)
        summary[m] = {
            "mean_ndcg5": float(arr.mean()),
            "std_ndcg5": float(arr.std()),
            "ci95_lo": lo,
            "ci95_hi": hi,
            "n_obs": int(len(arr)),
        }
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="42,137,271,314,1729")
    p.add_argument("--n_persons", type=int, default=40)
    p.add_argument("--n_docs", type=int, default=200)
    p.add_argument("--n_queries", type=int, default=120)
    p.add_argument("--gh_owner", type=str, default="vercel")
    p.add_argument("--gh_repo", type=str, default="next.js")
    p.add_argument("--gh_max_pages", type=int, default=4)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--out", type=str, default="data/results/phase2_master.json")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()

    # ---- Synthetic corpus: pool per-query observations across seeds ----
    print(f"[phase2] synthetic corpus across {len(seeds)} seeds")
    pooled: dict[str, list[float]] = {}
    for seed in seeds:
        corpus = make_synth_corpus(
            n_persons=args.n_persons, n_docs=args.n_docs,
            n_queries=args.n_queries, seed=seed,
        )
        per_method = _evaluate_all_methods(corpus, seed=seed, alpha=args.alpha)
        for m, vals in per_method.items():
            pooled.setdefault(m, []).extend(vals)
        print(f"  seed={seed} done")

    synth_summary = _summary_for_corpus("synth", pooled)
    synth_pairs = _compute_pair_stats(pooled, primary="scar-full")

    # ---- GitHub real corpus: single ingest, multi-seed bandit only ----
    print(f"[phase2] GitHub Discussions {args.gh_owner}/{args.gh_repo}")
    try:
        nodes = fetch_answered_discussions(args.gh_owner, args.gh_repo, max_pages=args.gh_max_pages)
        gh_corpus = discussions_to_corpus(nodes, name=f"{args.gh_owner}/{args.gh_repo}")
    except Exception as e:
        print(f"  [warn] gh fetch failed: {e}")
        gh_corpus = Corpus(name="empty", documents=(), queries=(), qrels=())

    gh_pooled: dict[str, list[float]] = {}
    if gh_corpus.documents and gh_corpus.queries:
        for seed in seeds:
            per_method = _evaluate_all_methods(gh_corpus, seed=seed, alpha=args.alpha)
            for m, vals in per_method.items():
                gh_pooled.setdefault(m, []).extend(vals)
        gh_summary = _summary_for_corpus("github", gh_pooled)
        gh_pairs = _compute_pair_stats(gh_pooled, primary="scar-full")
        # Also run pairs vs bm25 (since bm25 is strong on this corpus)
        gh_pairs_vs_bm25 = _compute_pair_stats(gh_pooled, primary="bm25")
    else:
        gh_summary = {}
        gh_pairs = []
        gh_pairs_vs_bm25 = []

    # ---- Print + save ----
    print(f"\n[phase2] SYNTHETIC CORPUS (n_obs={synth_summary[next(iter(synth_summary))]['n_obs']})")
    for m, v in synth_summary.items():
        print(f"  {m:<14} {v['mean_ndcg5']:.4f}  CI95=[{v['ci95_lo']:.4f}, {v['ci95_hi']:.4f}]")
    print(f"\n  vs scar-full   delta     p-value     Cohen's d  Holm")
    for r in synth_pairs:
        print(f"    {r['b']:<14} {r['delta_mean']:+.4f}    {r['p']:.4g}      {r['cohens_d']:+.3f}     {'YES' if r['holm_reject_at_005'] else 'no'}")

    print(f"\n[phase2] GITHUB CORPUS (n_obs={(gh_summary.get('scar-full') or {}).get('n_obs', 0)})")
    for m, v in gh_summary.items():
        print(f"  {m:<14} {v['mean_ndcg5']:.4f}  CI95=[{v['ci95_lo']:.4f}, {v['ci95_hi']:.4f}]")
    print(f"\n  vs scar-full   delta     p-value     Cohen's d  Holm")
    for r in gh_pairs:
        print(f"    {r['b']:<14} {r['delta_mean']:+.4f}    {r['p']:.4g}      {r['cohens_d']:+.3f}     {'YES' if r['holm_reject_at_005'] else 'no'}")
    print(f"\n  vs bm25        delta     p-value     Cohen's d  Holm")
    for r in gh_pairs_vs_bm25:
        print(f"    {r['b']:<14} {r['delta_mean']:+.4f}    {r['p']:.4g}      {r['cohens_d']:+.3f}     {'YES' if r['holm_reject_at_005'] else 'no'}")

    elapsed = time.time() - t0
    print(f"\n[phase2] total elapsed {elapsed:.2f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": vars(args),
        "elapsed_seconds": elapsed,
        "synthetic": {
            "summary": synth_summary,
            "pairs_vs_scar_full": synth_pairs,
        },
        "github": {
            "n_docs": len(gh_corpus.documents),
            "n_queries": len(gh_corpus.queries),
            "n_qrels": len(gh_corpus.qrels),
            "summary": gh_summary,
            "pairs_vs_scar_full": gh_pairs,
            "pairs_vs_bm25": gh_pairs_vs_bm25,
        },
    }, indent=2))
    print(f"[phase2] wrote {out_path}")


if __name__ == "__main__":
    main()
