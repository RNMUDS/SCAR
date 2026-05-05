"""Phase 1.1 smoke driver: run all 6 fixed arms + 3 LinUCB variants on the
synthetic corpus and dump a results JSON.

Usage (from ~/SCAR with venv active):
    python -m experiments.01_linucb.run_smoke --seed 42
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.baseline.linucb import (
    ARM_NAMES, make_linucb_base, make_linucb_warm, make_scar_linucb,
)
from src.sim.evaluator import evaluate_bandit, evaluate_fixed_arm
from src.sim.synth_corpus import make_synth_corpus
from src.sim.text_sim import BM25Backend


def _report_to_dict(rep) -> dict:
    return {
        "label": rep.label,
        "mean_ndcg5": rep.mean_ndcg5,
        "mean_mrr": rep.mean_mrr,
        "mean_recall5": rep.mean_recall5,
        "arm_pull_counts": rep.arm_pull_counts,
        "n_queries": len(rep.per_query),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_persons", type=int, default=30)
    parser.add_argument("--n_docs", type=int, default=80)
    parser.add_argument("--n_queries", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="data/results/01_linucb_smoke.json")
    args = parser.parse_args()

    print(f"[smoke] building corpus seed={args.seed} docs={args.n_docs} queries={args.n_queries}")
    corpus = make_synth_corpus(
        n_persons=args.n_persons,
        n_docs=args.n_docs,
        n_queries=args.n_queries,
        seed=args.seed,
    )
    bm25 = BM25Backend.from_corpus(corpus)

    results: dict[str, dict] = {}
    t0 = time.time()

    print("[smoke] evaluating fixed arms")
    for arm in ARM_NAMES:
        rep = evaluate_fixed_arm(corpus, arm_name=arm, bm25=bm25)
        results[arm] = _report_to_dict(rep)
        print(f"  {arm:<12} nDCG@5={rep.mean_ndcg5:.4f} MRR={rep.mean_mrr:.4f}")

    print("[smoke] evaluating bandit variants")
    for label, factory in [
        ("linucb-base", lambda: make_linucb_base(alpha=args.alpha, seed=args.seed)),
        ("linucb-warm", lambda: make_linucb_warm(alpha=args.alpha, seed=args.seed)),
        ("scar+linucb", lambda: make_scar_linucb(alpha=args.alpha, seed=args.seed)),
    ]:
        policy = factory()
        rep = evaluate_bandit(corpus, policy, bm25=bm25, label=label)
        results[label] = _report_to_dict(rep)
        print(f"  {label:<12} nDCG@5={rep.mean_ndcg5:.4f}  pulls={rep.arm_pull_counts}")

    elapsed = time.time() - t0
    print(f"[smoke] total elapsed: {elapsed:.2f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "config": {
                    "seed": args.seed,
                    "n_persons": args.n_persons,
                    "n_docs": args.n_docs,
                    "n_queries": args.n_queries,
                    "alpha": args.alpha,
                },
                "elapsed_seconds": elapsed,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"[smoke] wrote {out_path}")


if __name__ == "__main__":
    main()
