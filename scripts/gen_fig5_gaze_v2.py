"""Regenerated fig5: gaze method comparison at n=500 (5 seeds * 100 scenarios).
Replaces the original n=10 figure. Reads from data/results/02_gta_with_stats.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    src = Path("data/results/02_gta_with_stats.json")
    data = json.loads(src.read_text())
    summary = data["summary"]
    pair_stats = data["pair_stats"]

    methods = ["random", "frequency_only", "current_gaze_only", "gta"]
    means = [summary[m]["mean_ndcg5"] for m in methods]
    stds = [summary[m]["std_ndcg5"] for m in methods]
    ns = [summary[m]["n_obs"] for m in methods]

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 10})
    fig, ax = plt.subplots(figsize=(5.0, 2.8))

    x = np.arange(len(methods))
    colors = ["#BDBDBD", "#9CCC65", "#42A5F5", "#76B900"]
    bars = ax.bar(x, means, yerr=[s / np.sqrt(n) for s, n in zip(stds, ns)],
                  color=colors, edgecolor="white", capsize=4, zorder=3)

    for i, (m, mean) in enumerate(zip(methods, means)):
        ax.text(i, mean + 0.025, f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["Random", "Frequency-Only", "Current-Gaze-Only", "GTA (ours)"], rotation=15, ha="right")
    ax.set_ylabel("nDCG@5")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate Holm-significant comparisons
    significance_lines = []
    for ps in pair_stats:
        if ps.get("holm_reject_at_005"):
            significance_lines.append(f"{ps['a']} > {ps['b']}: p={ps['p']:.2g}, d={ps['cohens_d']:+.2f}")
    title = f"Gaze trajectory methods (n={ns[0]} pooled across 5 seeds)"
    ax.set_title(title, fontsize=10)

    fig.tight_layout()
    out_dir = Path("data/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig5_gaze_v2.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig5_gaze_v2.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("fig5_v2 done")


if __name__ == "__main__":
    main()
