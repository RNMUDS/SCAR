"""fig16: noise sweep — nDCG@5 vs sigma for BM25, SCAR, LinUCB.
Reads data/results/04_noise_sweep.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    src = Path("data/results/04_noise_sweep.json")
    data = json.loads(src.read_text())
    sigmas = data["sigmas"]
    results = data["results"]

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 10})
    fig, ax = plt.subplots(figsize=(5.0, 3.2))

    style = {
        "bm25":        ("o", "#9E9E9E", "BM25 (text only)"),
        "scar-full":   ("s", "#76B900", "SCAR (fixed weights)"),
    }
    for method, (marker, color, label) in style.items():
        if method not in results:
            continue
        pts = results[method]
        x = [r["sigma"] for r in pts]
        y = [r["mean_ndcg5"] for r in pts]
        ax.plot(x, y, marker=marker, color=color, linewidth=1.8, markersize=7,
                markerfacecolor=color, markeredgecolor="white", label=label)

    ax.set_xlabel(r"Noise level $\sigma$ on proximity/gaze")
    ax.set_ylabel("nDCG@5")
    ax.set_title("Robustness to spatial sensor noise")
    ax.set_xticks(sigmas)
    ax.set_xticklabels([f"{s:.1f}" for s in sigmas])
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_dir = Path("data/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig16_noise_sweep.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig16_noise_sweep.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("fig16 done")


if __name__ == "__main__":
    main()
