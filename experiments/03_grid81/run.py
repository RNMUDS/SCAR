"""Phase 1.3 + 2.3 driver: 81-point LHS grid search over the SCAR weight simplex.

For each weight tuple, evaluate mean nDCG@5 on the synthetic corpus (later
swapped for the GitHub real corpus once it's ingested).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.grid.lhs_search import lhs_simplex_5d
from src.sim.evaluator import evaluate_fixed_arm
from src.sim.scar_scorer import ScarWeights, score
from src.sim.synth_corpus import make_synth_corpus
from src.sim.text_sim import BM25Backend
from src.utils.metrics import ndcg_at_k_with_qrels
from src.utils.types import Corpus


def _eval_weights(corpus: Corpus, bm25: BM25Backend, weights: ScarWeights) -> float:
    """Mean nDCG@5 with the given weight tuple, fixed-arm style."""
    qrels_per_query = corpus.relevance_map()
    per_q_ndcg: list[float] = []
    for q in corpus.queries:
        bm25_scores = bm25.scores(q.text)
        scored = []
        for d in corpus.documents:
            ts = bm25_scores.get(d.doc_id, 0.0)
            r = score(ts, d, q.context, weights)
            scored.append(r)
        scored.sort(key=lambda r: r.final_score, reverse=True)
        ranked = [r.doc_id for r in scored]
        per_q_ndcg.append(ndcg_at_k_with_qrels(ranked, qrels_per_query.get(q.query_id, {}), k=5))
    return float(np.mean(per_q_ndcg))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_points", type=int, default=81)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_persons", type=int, default=40)
    p.add_argument("--n_docs", type=int, default=200)
    p.add_argument("--n_queries", type=int, default=120)
    p.add_argument("--out", type=str, default="data/results/03_grid81.json")
    args = p.parse_args()

    print(f"[grid] building corpus")
    corpus = make_synth_corpus(
        n_persons=args.n_persons,
        n_docs=args.n_docs,
        n_queries=args.n_queries,
        seed=args.seed,
    )
    bm25 = BM25Backend.from_corpus(corpus)

    print(f"[grid] sampling {args.n_points} points on the 5-simplex (LHS)")
    points = lhs_simplex_5d(n_points=args.n_points, seed=args.seed)

    t0 = time.time()
    scored: list[dict] = []
    for i, w in enumerate(points):
        s = _eval_weights(corpus, bm25, w)
        scored.append({
            "alpha": w.alpha, "beta": w.beta, "gamma": w.gamma,
            "delta": w.delta, "epsilon": w.epsilon,
            "ndcg5": s,
        })
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  point {i + 1}/{len(points)}  nDCG@5={s:.4f}")
    elapsed = time.time() - t0

    best = max(scored, key=lambda r: r["ndcg5"])
    paper_default = scored[0]  # always first
    print(f"\n[grid] paper default: nDCG@5={paper_default['ndcg5']:.4f}")
    print(f"[grid] best on grid : nDCG@5={best['ndcg5']:.4f} at "
          f"alpha={best['alpha']:.2f} beta={best['beta']:.2f} gamma={best['gamma']:.2f} "
          f"delta={best['delta']:.2f} epsilon={best['epsilon']:.2f}")
    print(f"[grid] elapsed: {elapsed:.2f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": vars(args),
        "elapsed_seconds": elapsed,
        "n_points": args.n_points,
        "paper_default": paper_default,
        "best_point": best,
        "all_points": scored,
    }, indent=2))
    print(f"[grid] wrote {out_path}")


if __name__ == "__main__":
    main()
