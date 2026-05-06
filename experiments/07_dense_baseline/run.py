"""Phase 7 driver: dense-retrieval baseline (nomic-embed-text via Ollama).

Compares 4 conditions on the synthetic simulator corpus to disentangle
"text-matching method" from "spatial signals":

    BM25  / BM25 + SCAR-full
    Dense / Dense + SCAR-full

Reads embeddings from Ollama's /api/embeddings endpoint
(model: nomic-embed-text, 768-dim). Documents are embedded once at start;
queries are embedded on demand.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.sim.evaluator import evaluate_fixed_arm
from src.sim.synth_corpus import make_synth_corpus
from src.sim.text_sim import BM25Backend, DenseBackend


CONDITIONS = (
    ("bm25",         "bm25",       "BM25 only"),
    ("bm25+scar",    "scar-full",  "BM25 + SCAR-full"),
    ("dense",        "bm25",       "Dense only"),
    ("dense+scar",   "scar-full",  "Dense + SCAR-full"),
)


def _per_query_scores(report) -> list[float]:
    return [r.ndcg5 for r in report.per_query]


def _paired_diff_stats(a: list[float], b: list[float]) -> dict[str, float]:
    """Mean of paired differences (b - a) and standard error."""
    arr = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean_diff": 0.0, "se": 0.0}
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {"n": n, "mean_diff": mean, "se": se}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_persons", type=int, default=40)
    p.add_argument("--n_docs", type=int, default=200)
    p.add_argument("--n_queries", type=int, default=120)
    p.add_argument("--ollama_url", type=str, default="http://localhost:11434")
    p.add_argument("--model", type=str, default="nomic-embed-text")
    p.add_argument("--out", type=str, default="data/results/07_dense_baseline.json")
    args = p.parse_args()

    print(f"[07] building corpus seed={args.seed} n_docs={args.n_docs} n_queries={args.n_queries}")
    corpus = make_synth_corpus(
        n_persons=args.n_persons,
        n_docs=args.n_docs,
        n_queries=args.n_queries,
        seed=args.seed,
    )

    print(f"[07] building BM25 backend")
    bm25 = BM25Backend.from_corpus(corpus)

    t0 = time.time()
    print(f"[07] embedding {args.n_docs} docs with {args.model} via {args.ollama_url}")
    dense = DenseBackend.from_corpus(corpus, model=args.model, url=args.ollama_url)
    t_embed = time.time() - t0
    print(f"[07] doc embeddings ready ({t_embed:.1f}s)")

    per_query: dict[str, list[float]] = {}
    summary: dict[str, dict] = {}

    for cond, arm, label in CONDITIONS:
        backend = bm25 if cond.startswith("bm25") else dense
        rep = evaluate_fixed_arm(corpus, arm, bm25=backend)
        per_query[cond] = _per_query_scores(rep)
        summary[cond] = {
            "label": label,
            "arm": arm,
            "backend": "bm25" if cond.startswith("bm25") else "dense",
            "mean_ndcg5": rep.mean_ndcg5,
            "mean_mrr": rep.mean_mrr,
            "mean_recall5": rep.mean_recall5,
        }
        print(f"  {cond:<12} nDCG@5={rep.mean_ndcg5:.4f} MRR={rep.mean_mrr:.4f} R@5={rep.mean_recall5:.4f}")

    # paired differences (within-query, same corpus): isolate spatial-bonus effect
    deltas = {
        "bm25_spatial_gain": _paired_diff_stats(per_query["bm25"], per_query["bm25+scar"]),
        "dense_spatial_gain": _paired_diff_stats(per_query["dense"], per_query["dense+scar"]),
        "text_method_gain_no_spatial": _paired_diff_stats(per_query["bm25"], per_query["dense"]),
        "text_method_gain_with_spatial": _paired_diff_stats(per_query["bm25+scar"], per_query["dense+scar"]),
    }
    print()
    print("[07] paired Delta nDCG@5 (mean +/- SE):")
    for k, v in deltas.items():
        print(f"  {k:<32} {v['mean_diff']:+.4f} +/- {v['se']:.4f}  (n={v['n']})")

    elapsed = time.time() - t0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": vars(args),
        "summary": summary,
        "deltas": deltas,
        "elapsed_seconds": elapsed,
        "doc_embedding_seconds": t_embed,
    }, indent=2))
    print(f"\n[07] wrote {out_path}")


if __name__ == "__main__":
    main()
