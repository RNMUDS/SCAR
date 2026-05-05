"""Phase 1.4 + 2.4 driver: noise sweep over sigma in {0, 0.1, 0.2, 0.3, 0.5}.

For each sigma:
    - corrupt every query's spatial context
    - evaluate BM25, scar-full, linucb-base on corrupted contexts
    - compare against ground-truth qrels (which use the *true* context)

This tests robustness: SCAR should degrade gracefully; LinUCB should auto-discount
noisy spatial features.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.baseline.linucb import make_linucb_base
from src.noise.inject import corrupt_query
from src.sim.evaluator import evaluate_bandit, evaluate_fixed_arm
from src.sim.synth_corpus import make_synth_corpus
from src.sim.text_sim import BM25Backend
from src.utils.types import Corpus


def _all_persons(corpus: Corpus) -> list[str]:
    s = set()
    for d in corpus.documents:
        s.add(d.author)
        for m in d.mentions:
            s.add(m)
    for q in corpus.queries:
        s.add(q.context.user_id)
        for p in q.context.nearby_persons:
            s.add(p)
        if q.context.gaze_target:
            s.add(q.context.gaze_target)
    return sorted(s)


def _corrupt_corpus(corpus: Corpus, sigma: float, seed: int) -> Corpus:
    """Return a copy with all query contexts corrupted at level sigma."""
    pool = _all_persons(corpus)
    rng = random.Random(seed)
    new_queries = tuple(corrupt_query(q, sigma, pool, rng) for q in corpus.queries)
    return replace(corpus, queries=new_queries)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_docs", type=int, default=200)
    p.add_argument("--n_queries", type=int, default=120)
    p.add_argument("--sigmas", type=str, default="0.0,0.1,0.2,0.3,0.5")
    p.add_argument("--out", type=str, default="data/results/04_noise_sweep.json")
    args = p.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",")]

    print(f"[noise] building base corpus")
    corpus = make_synth_corpus(
        n_persons=40, n_docs=args.n_docs, n_queries=args.n_queries, seed=args.seed
    )

    results: dict[str, list[dict]] = {"bm25": [], "scar-full": [], "linucb-base": []}

    t0 = time.time()
    for sigma in sigmas:
        print(f"[noise] sigma={sigma}")
        corrupted = _corrupt_corpus(corpus, sigma, seed=args.seed + int(sigma * 100))
        bm25 = BM25Backend.from_corpus(corrupted)

        bm25_rep = evaluate_fixed_arm(corrupted, "bm25", bm25=bm25)
        scar_rep = evaluate_fixed_arm(corrupted, "scar-full", bm25=bm25)
        bandit_rep = evaluate_bandit(
            corrupted, make_linucb_base(alpha=1.0, seed=args.seed),
            bm25=bm25, label="linucb-base",
        )

        for k, rep in [("bm25", bm25_rep), ("scar-full", scar_rep), ("linucb-base", bandit_rep)]:
            results[k].append({
                "sigma": sigma,
                "mean_ndcg5": rep.mean_ndcg5,
                "mean_mrr": rep.mean_mrr,
                "mean_recall5": rep.mean_recall5,
            })
            print(f"  {k:<14} nDCG@5={rep.mean_ndcg5:.4f}")

    elapsed = time.time() - t0
    print(f"\n[noise] elapsed {elapsed:.2f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": vars(args),
        "sigmas": sigmas,
        "results": results,
        "elapsed_seconds": elapsed,
    }, indent=2))
    print(f"[noise] wrote {out_path}")


if __name__ == "__main__":
    main()
