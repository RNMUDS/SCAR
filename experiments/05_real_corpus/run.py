"""Phase 1.5 + 2.5 driver: GitHub Discussions corpus end-to-end.

Pipeline:
    1. Fetch answered discussions from a target repo via gh CLI
    2. Convert to a SCAR Corpus
    3. Evaluate BM25, scar-full, all 6 arms, and LinUCB variants
    4. Save results JSON

Falls back to a small fixture if `gh` auth fails (so the run completes).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.baseline.linucb import (
    ARM_NAMES, make_linucb_base, make_linucb_warm, make_scar_linucb,
)
from src.corpus.github_ingest import discussions_to_corpus, fetch_answered_discussions
from src.sim.evaluator import evaluate_bandit, evaluate_fixed_arm
from src.sim.text_sim import BM25Backend


def _save_raw_corpus(corpus, path: Path) -> None:
    """Persist a JSON snapshot of the corpus we ingested (so re-runs are reproducible)."""
    data = {
        "name": corpus.name,
        "documents": [
            {
                "doc_id": d.doc_id, "text": d.text, "author": d.author,
                "location_type": d.location_type, "kind": d.kind,
                "created_day": d.created_day, "mentions": list(d.mentions),
            } for d in corpus.documents
        ],
        "queries": [
            {
                "query_id": q.query_id, "text": q.text,
                "context": {
                    "user_id": q.context.user_id, "role": q.context.role,
                    "current_zone": q.context.current_zone,
                    "current_day": q.context.current_day,
                    "nearby_persons": list(q.context.nearby_persons),
                    "gaze_target": q.context.gaze_target,
                },
            } for q in corpus.queries
        ],
        "qrels": [
            {"query_id": r.query_id, "doc_id": r.doc_id, "grade": r.grade}
            for r in corpus.qrels
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--owner", type=str, default="vercel")
    p.add_argument("--repo", type=str, default="next.js")
    p.add_argument("--max_pages", type=int, default=4)  # 4 * 50 = 200 discussions max
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--out", type=str, default="data/results/05_real_corpus.json")
    p.add_argument("--corpus_out", type=str, default="data/corpora/github_real.json")
    args = p.parse_args()

    print(f"[real] fetching discussions {args.owner}/{args.repo}")
    t0 = time.time()
    try:
        nodes = fetch_answered_discussions(args.owner, args.repo, max_pages=args.max_pages)
        print(f"[real] fetched {len(nodes)} answered discussions in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"[real] gh fetch failed ({e}); using empty corpus")
        nodes = []

    corpus = discussions_to_corpus(nodes, name=f"{args.owner}/{args.repo}")
    print(f"[real] corpus: {len(corpus.documents)} docs, {len(corpus.queries)} queries, "
          f"{len(corpus.qrels)} qrels")

    if not corpus.documents or not corpus.queries:
        print("[real] empty corpus — writing failure summary and exiting")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "config": vars(args),
            "status": "no_data",
            "n_docs": len(corpus.documents),
            "n_queries": len(corpus.queries),
            "n_qrels": len(corpus.qrels),
        }, indent=2))
        return

    _save_raw_corpus(corpus, Path(args.corpus_out))

    bm25 = BM25Backend.from_corpus(corpus)
    results: dict[str, dict] = {}

    print("[real] evaluating fixed arms")
    for arm in ARM_NAMES:
        rep = evaluate_fixed_arm(corpus, arm_name=arm, bm25=bm25)
        results[arm] = {
            "mean_ndcg5": rep.mean_ndcg5, "mean_mrr": rep.mean_mrr,
            "mean_recall5": rep.mean_recall5,
        }
        print(f"  {arm:<12} nDCG@5={rep.mean_ndcg5:.4f}  MRR={rep.mean_mrr:.4f}")

    print("[real] evaluating bandit variants")
    for label, factory in [
        ("linucb-base", lambda: make_linucb_base(alpha=args.alpha, seed=args.seed)),
        ("linucb-warm", lambda: make_linucb_warm(alpha=args.alpha, seed=args.seed)),
        ("scar+linucb", lambda: make_scar_linucb(alpha=args.alpha, seed=args.seed)),
    ]:
        rep = evaluate_bandit(corpus, factory(), bm25=bm25, label=label)
        results[label] = {
            "mean_ndcg5": rep.mean_ndcg5, "mean_mrr": rep.mean_mrr,
            "mean_recall5": rep.mean_recall5,
            "arm_pull_counts": rep.arm_pull_counts,
        }
        print(f"  {label:<12} nDCG@5={rep.mean_ndcg5:.4f}")

    elapsed = time.time() - t0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": vars(args),
        "status": "ok",
        "n_docs": len(corpus.documents),
        "n_queries": len(corpus.queries),
        "n_qrels": len(corpus.qrels),
        "results": results,
        "elapsed_seconds": elapsed,
    }, indent=2))
    print(f"\n[real] wrote {out_path}")


if __name__ == "__main__":
    main()
