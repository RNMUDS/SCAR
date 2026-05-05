"""fig17: per-arm performance on real GitHub Discussions corpus.
Reads data/results/phase2_master.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    src = Path("data/results/phase2_master.json")
    data = json.loads(src.read_text())
    gh = data["github"]
    if not gh.get("summary"):
        print("fig17: no github data")
        return

    summ = gh["summary"]
    pairs_vs_bm25 = gh.get("pairs_vs_bm25", [])
    sig_set = {p["b"] for p in pairs_vs_bm25 if p.get("holm_reject_at_005")}

    methods = ["bm25", "bm25+prox", "bm25+gaze", "bm25+loc", "bm25+time",
               "scar-full", "linucb-base", "linucb-warm", "scar+linucb"]
    means = [summ[m]["mean_ndcg5"] for m in methods]
    ci_lo = [summ[m]["ci95_lo"] for m in methods]
    ci_hi = [summ[m]["ci95_hi"] for m in methods]
    yerr = [[m - lo for m, lo in zip(means, ci_lo)], [hi - m for m, hi in zip(means, ci_hi)]]

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 10})
    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    bm25_idx = methods.index("bm25")
    bm25_mean = means[bm25_idx]
    colors = []
    for i, m in enumerate(methods):
        if m == "bm25":
            colors.append("#757575")
        elif means[i] > bm25_mean:
            colors.append("#76B900" if m in sig_set else "#C5E1A5")
        else:
            colors.append("#EF5350" if m in sig_set else "#FFCDD2")

    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=yerr, color=colors, edgecolor="white", capsize=2.5, zorder=3)
    ax.axhline(bm25_mean, color="black", linestyle="--", linewidth=0.8, zorder=2,
               label=f"BM25 baseline = {bm25_mean:.4f}")

    for i, m in enumerate(means):
        offset = max([yerr[1][i], 0.0008])
        ax.text(i, m + offset + 0.0005, f"{m:.4f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("nDCG@5")
    ax.set_ylim(0.965, 0.990)
    ax.set_title(f"Real corpus (vercel/next.js Discussions, n_obs={summ['bm25']['n_obs']})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_dir = Path("data/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig17_real_corpus.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig17_real_corpus.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("fig17 done")


if __name__ == "__main__":
    main()
