"""fig15: 5D weight simplex grid search heatmap (2D projections).
Reads data/results/03_grid81.json. Shows nDCG@5 as a function of (alpha, beta+gamma+delta+epsilon).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    src = Path("data/results/03_grid81.json")
    data = json.loads(src.read_text())
    points = data["all_points"]

    alpha = np.array([p["alpha"] for p in points])
    beta_sum = np.array([p["beta"] + p["gamma"] + p["delta"] + p["epsilon"] for p in points])
    ndcg = np.array([p["ndcg5"] for p in points])

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 10})
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    sc = ax.scatter(alpha, beta_sum, c=ndcg, cmap="viridis", s=50, edgecolor="white", linewidth=0.5)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("nDCG@5")

    # mark default and best
    default = data["paper_default"]
    best = data["best_point"]
    ax.scatter([default["alpha"]], [default["beta"] + default["gamma"] + default["delta"] + default["epsilon"]],
               s=160, marker="*", color="red", edgecolor="black", linewidth=1, zorder=5,
               label=f"paper default (nDCG={default['ndcg5']:.3f})")
    ax.scatter([best["alpha"]], [best["beta"] + best["gamma"] + best["delta"] + best["epsilon"]],
               s=200, marker="X", color="orange", edgecolor="black", linewidth=1, zorder=4,
               label=f"grid best (nDCG={best['ndcg5']:.3f})")

    ax.set_xlabel(r"$\alpha$ (text weight)")
    ax.set_ylabel(r"$\beta + \gamma + \delta + \varepsilon$ (spatial weight sum)")
    ax.set_title(f"5-simplex LHS grid ({len(points)} points)")
    ax.legend(loc="lower left", fontsize=7)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_dir = Path("data/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig15_grid_simplex.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig15_grid_simplex.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("fig15 done")


if __name__ == "__main__":
    main()
