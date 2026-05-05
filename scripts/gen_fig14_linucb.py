"""fig14: LinUCB vs SCAR vs BM25 comparison on synth + GitHub corpora.
Reads phase2_master.json.
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

    methods = ["bm25", "bm25+prox", "scar-full", "linucb-base", "linucb-warm", "scar+linucb"]
    synth = data["synthetic"]["summary"]
    gh = data["github"]["summary"]

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=False)

    for ax, summ, title, ylim in [
        (axes[0], synth, f"Synthetic corpus (n_obs={summ_or_zero(synth)})", (0.7, 1.05)),
        (axes[1], gh,    f"GitHub Discussions corpus (n_obs={summ_or_zero(gh)})", (0.96, 0.99)),
    ]:
        if not summ:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        means = [summ[m]["mean_ndcg5"] for m in methods]
        ci_lo = [summ[m]["ci95_lo"] for m in methods]
        ci_hi = [summ[m]["ci95_hi"] for m in methods]
        yerr = [
            [m - lo for m, lo in zip(means, ci_lo)],
            [hi - m for m, hi in zip(means, ci_hi)],
        ]
        x = np.arange(len(methods))
        colors = ["#BDBDBD", "#9CCC65", "#76B900", "#42A5F5", "#1E88E5", "#0D47A1"]
        ax.bar(x, means, yerr=yerr, color=colors, edgecolor="white", capsize=3, zorder=3)
        for i, mean in enumerate(means):
            ax.text(i, mean + (ylim[1] - ylim[0]) * 0.03, f"{mean:.3f}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_ylim(*ylim)
        ax.set_ylabel("nDCG@5 (CI95)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_dir = Path("data/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig14_linucb_comparison.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig14_linucb_comparison.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("fig14 done")


def summ_or_zero(s):
    if not s:
        return 0
    first = next(iter(s.values()))
    return first.get("n_obs", 0)


if __name__ == "__main__":
    main()
