"""Phase 6 driver: proximity-only weight sweep.

Tests whether the proximity feature alone (text + proximity, all other spatial
weights zeroed) carries the entire generalisable spatial signal. We sweep
beta in {0.05, 0.20, 0.50, 0.80}, fixing alpha = 1 - beta, gamma = delta =
epsilon = 0, and evaluate on both the synthetic corpus and the GitHub
Discussions real corpus. Multi-seed bootstrap CI per cell.

Outputs data/results/06_proximity_sweep.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.corpus.github_ingest import discussions_to_corpus, fetch_answered_discussions
from src.sim.scar_scorer import ScarWeights, score
from src.sim.synth_corpus import make_synth_corpus
from src.sim.text_sim import BM25Backend
from src.utils.metrics import ndcg_at_k_with_qrels
from src.utils.stats import bootstrap_mean_ci, cohens_d_paired, paired_ttest_p
from src.utils.types import Corpus


def _eval_weights(corpus: Corpus, bm25: BM25Backend, w: ScarWeights) -> list[float]:
    """Per-query nDCG@5 list under the given weights (fixed-arm)."""
    qrels_per_query = corpus.relevance_map()
    out: list[float] = []
    for q in corpus.queries:
        bm25_scores = bm25.scores(q.text)
        scored = []
        for d in corpus.documents:
            ts = bm25_scores.get(d.doc_id, 0.0)
            scored.append(score(ts, d, q.context, w))
        scored.sort(key=lambda r: r.final_score, reverse=True)
        ranked = [r.doc_id for r in scored]
        out.append(ndcg_at_k_with_qrels(ranked, qrels_per_query.get(q.query_id, {}), k=5))
    return out


def _proximity_only_weights(beta: float) -> ScarWeights:
    """Proximity-only: alpha = 1 - beta, beta = beta, others = 0."""
    return ScarWeights(alpha=1.0 - beta, beta=beta, gamma=0.0, delta=0.0, epsilon=0.0)


def _bm25_baseline_weights() -> ScarWeights:
    return ScarWeights(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, epsilon=0.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="42,137,271,314,1729")
    p.add_argument("--betas", type=str, default="0.05,0.20,0.50,0.80")
    p.add_argument("--n_persons", type=int, default=40)
    p.add_argument("--n_docs", type=int, default=200)
    p.add_argument("--n_queries", type=int, default=120)
    p.add_argument("--gh_owner", type=str, default="vercel")
    p.add_argument("--gh_repo", type=str, default="next.js")
    p.add_argument("--gh_max_pages", type=int, default=4)
    p.add_argument("--out", type=str, default="data/results/06_proximity_sweep.json")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    betas = [float(b) for b in args.betas.split(",")]

    out: dict[str, dict] = {"synthetic": {}, "github": {}}
    t0 = time.time()

    # ---- Synthetic corpus ----
    print(f"[prox] synthetic ({len(seeds)} seeds x {len(betas)} betas)")
    synth_bm25_pool: list[float] = []
    synth_per_beta: dict[float, list[float]] = {b: [] for b in betas}
    for seed in seeds:
        corpus = make_synth_corpus(
            n_persons=args.n_persons, n_docs=args.n_docs,
            n_queries=args.n_queries, seed=seed,
        )
        bm25 = BM25Backend.from_corpus(corpus)
        bm25_vals = _eval_weights(corpus, bm25, _bm25_baseline_weights())
        synth_bm25_pool.extend(bm25_vals)
        for beta in betas:
            vals = _eval_weights(corpus, bm25, _proximity_only_weights(beta))
            synth_per_beta[beta].extend(vals)
        print(f"  seed={seed} done")

    out["synthetic"]["bm25"] = _summarize(synth_bm25_pool)
    out["synthetic"]["per_beta"] = {}
    for beta in betas:
        vals = synth_per_beta[beta]
        s = _summarize(vals)
        # paired vs bm25 baseline
        s["paired_p_vs_bm25"] = paired_ttest_p(vals, synth_bm25_pool)
        s["cohens_d_vs_bm25"] = cohens_d_paired(vals, synth_bm25_pool)
        out["synthetic"]["per_beta"][f"{beta:.2f}"] = s

    # ---- GitHub real corpus ----
    print(f"[prox] github real ({args.gh_owner}/{args.gh_repo})")
    try:
        nodes = fetch_answered_discussions(args.gh_owner, args.gh_repo, max_pages=args.gh_max_pages)
        gh_corpus = discussions_to_corpus(nodes, name=f"{args.gh_owner}/{args.gh_repo}")
    except Exception as e:
        print(f"  [warn] gh fetch failed: {e}")
        gh_corpus = Corpus(name="empty", documents=(), queries=(), qrels=())

    if gh_corpus.documents and gh_corpus.queries:
        gh_bm25 = BM25Backend.from_corpus(gh_corpus)
        gh_bm25_pool: list[float] = []
        gh_per_beta: dict[float, list[float]] = {b: [] for b in betas}
        for seed in seeds:
            # GitHub corpus is fixed; only the per-beta evaluation reuses it.
            # We pool seeds for stat consistency with the synthetic pipeline.
            bm25_vals = _eval_weights(gh_corpus, gh_bm25, _bm25_baseline_weights())
            gh_bm25_pool.extend(bm25_vals)
            for beta in betas:
                vals = _eval_weights(gh_corpus, gh_bm25, _proximity_only_weights(beta))
                gh_per_beta[beta].extend(vals)

        out["github"]["bm25"] = _summarize(gh_bm25_pool)
        out["github"]["per_beta"] = {}
        for beta in betas:
            vals = gh_per_beta[beta]
            s = _summarize(vals)
            s["paired_p_vs_bm25"] = paired_ttest_p(vals, gh_bm25_pool)
            s["cohens_d_vs_bm25"] = cohens_d_paired(vals, gh_bm25_pool)
            out["github"]["per_beta"][f"{beta:.2f}"] = s
        out["github"]["n_docs"] = len(gh_corpus.documents)
        out["github"]["n_queries"] = len(gh_corpus.queries)

    elapsed = time.time() - t0
    out["config"] = vars(args)
    out["elapsed_seconds"] = elapsed

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    # ---- Print summary ----
    print(f"\n[prox] SYNTHETIC corpus")
    print(f"  bm25-only         {out['synthetic']['bm25']['mean']:.4f}  CI95={fmt_ci(out['synthetic']['bm25'])}")
    for b in betas:
        s = out['synthetic']['per_beta'][f"{b:.2f}"]
        print(f"  beta={b:.2f}        {s['mean']:.4f}  CI95={fmt_ci(s)}  d={s['cohens_d_vs_bm25']:+.3f}  p={s['paired_p_vs_bm25']:.3g}")

    if "per_beta" in out["github"]:
        print(f"\n[prox] GITHUB real corpus (n_docs={out['github']['n_docs']}, n_queries={out['github']['n_queries']})")
        print(f"  bm25-only         {out['github']['bm25']['mean']:.4f}  CI95={fmt_ci(out['github']['bm25'])}")
        for b in betas:
            s = out['github']['per_beta'][f"{b:.2f}"]
            print(f"  beta={b:.2f}        {s['mean']:.4f}  CI95={fmt_ci(s)}  d={s['cohens_d_vs_bm25']:+.3f}  p={s['paired_p_vs_bm25']:.3g}")

    print(f"\n[prox] elapsed {elapsed:.2f}s; wrote {out_path}")


def _summarize(values: list[float]) -> dict:
    arr = np.asarray(values)
    if len(arr) == 0:
        return {"mean": 0.0, "ci95_lo": 0.0, "ci95_hi": 0.0, "n": 0}
    mean, lo, hi = bootstrap_mean_ci(arr, n_resamples=1000, seed=0)
    return {"mean": float(mean), "ci95_lo": lo, "ci95_hi": hi, "n": int(len(arr))}


def fmt_ci(s: dict) -> str:
    return f"[{s['ci95_lo']:.4f}, {s['ci95_hi']:.4f}]"


if __name__ == "__main__":
    main()
