"""fig18: proximity-only beta sweep on synthetic and GitHub real corpora.

Reads data/results/06_proximity_sweep.json. Produces a 2-panel side-by-side
figure showing the saturation curve and the 'ceiling corridor' where proximity
weight is useful.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    src = Path("data/results/06_proximity_sweep.json")
    data = json.loads(src.read_text())

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))

    panels = [
        (axes[0], data["synthetic"], "Synthetic corpus", (0.55, 1.05)),
        (axes[1], data["github"], "GitHub Discussions (real)", (0.58, 1.00)),
    ]

    for ax, side, title, ylim in panels:
        if "per_beta" not in side:
            ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        beta_keys = sorted(side["per_beta"].keys(), key=lambda s: float(s))
        betas = [float(b) for b in beta_keys]
        means = [side["per_beta"][b]["mean"] for b in beta_keys]
        ci_lo = [side["per_beta"][b]["ci95_lo"] for b in beta_keys]
        ci_hi = [side["per_beta"][b]["ci95_hi"] for b in beta_keys]
        yerr = [[m - lo for m, lo in zip(means, ci_lo)], [hi - m for m, hi in zip(means, ci_hi)]]

        # BM25 baseline
        bm25_mean = side["bm25"]["mean"]
        ax.axhline(bm25_mean, color="black", linestyle="--", linewidth=0.8, zorder=2,
                   label=f"BM25 baseline = {bm25_mean:.3f}")

        # proximity-only curve
        ax.errorbar(betas, means, yerr=yerr, marker="o", color="#76B900", linewidth=1.8,
                    markersize=8, markerfacecolor="#76B900", markeredgecolor="white",
                    capsize=4, zorder=3, label="text + proximity only")

        # Highlight β=0.20 (paper default)
        idx_20 = beta_keys.index("0.20")
        ax.scatter([0.20], [means[idx_20]], s=200, marker="*", color="orange",
                   edgecolor="black", linewidth=1, zorder=5, label=r"$\beta{=}0.20$ (paper default)")

        # Annotate values
        for b, m in zip(betas, means):
            ax.annotate(f"{m:.3f}", xy=(b, m), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7, color="#333")

        ax.set_xlabel(r"Proximity weight $\beta$ (text weight $\alpha = 1 - \beta$)")
        ax.set_ylabel("nDCG@5")
        ax.set_title(title)
        ax.set_xticks(betas)
        ax.set_xticklabels([f"{b:.2f}" for b in betas])
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.3, zorder=0)
        ax.legend(loc="lower left", fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_dir = Path("data/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig18_proximity_sweep.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig18_proximity_sweep.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("fig18 done")


if __name__ == "__main__":
    main()
