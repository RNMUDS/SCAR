"""fig14: LinUCB vs SCAR vs BM25 comparison on synth + GitHub corpora.

Style follows fig17: per-bar colour encodes significance vs BM25
(green = significant gain, light green = gain n.s., red = significant
loss, pink = loss n.s., grey = BM25 baseline). A dashed BM25 reference
line and value labels positioned just above the upper whisker keep the
chart readable. Reads ``data/results/phase2_master.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    "bm25",
    "bm25+prox",
    "scar-full",
    "linucb-base",
    "linucb-warm",
    "scar+linucb",
]


def _bar_colours(
    methods: list[str],
    means: list[float],
    bm25_mean: float,
    sig_set: set[str],
) -> list[str]:
    colours: list[str] = []
    for name, mean in zip(methods, means):
        if name == "bm25":
            colours.append("#757575")
        elif mean > bm25_mean:
            colours.append("#76B900" if name in sig_set else "#C5E1A5")
        else:
            colours.append("#EF5350" if name in sig_set else "#FFCDD2")
    return colours


def _draw_panel(
    ax,
    summary: dict,
    pairs_vs_bm25: Iterable[dict],
    title: str,
    ylim: tuple[float, float],
    *,
    label_fmt: str,
    methods: list[str] = METHODS,
) -> None:
    if not summary:
        ax.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return

    sig_set = {p["b"] for p in pairs_vs_bm25 if p.get("holm_reject_at_005")}
    means = [summary[m]["mean_ndcg5"] for m in methods]
    ci_lo = [summary[m]["ci95_lo"] for m in methods]
    ci_hi = [summary[m]["ci95_hi"] for m in methods]
    yerr = [
        [m - lo for m, lo in zip(means, ci_lo)],
        [hi - m for m, hi in zip(means, ci_hi)],
    ]

    bm25_mean = summary["bm25"]["mean_ndcg5"]
    colours = _bar_colours(methods, means, bm25_mean, sig_set)

    x = np.arange(len(methods))
    ax.bar(
        x,
        means,
        yerr=yerr,
        color=colours,
        edgecolor="white",
        capsize=2.5,
        zorder=3,
    )
    ax.axhline(
        bm25_mean,
        color="black",
        linestyle="--",
        linewidth=0.8,
        zorder=2,
        label=f"BM25 baseline = {label_fmt.format(bm25_mean)}",
    )

    offset = max((ylim[1] - ylim[0]) * 0.012, 0.0008)
    for i, (mean, hi) in enumerate(zip(means, ci_hi)):
        ax.text(
            i,
            hi + offset,
            label_fmt.format(mean),
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylim(*ylim)
    ax.set_ylabel("nDCG@5")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _n_obs(summary: dict | None) -> int:
    if not summary:
        return 0
    first = next(iter(summary.values()))
    return first.get("n_obs", 0)


def main() -> None:
    src = Path("data/results/phase2_master.json")
    data = json.loads(src.read_text())

    synth = data["synthetic"]
    gh = data["github"]

    plt.rcParams.update(
        {"font.family": "serif", "font.size": 9, "axes.labelsize": 10}
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=False)

    _draw_panel(
        axes[0],
        synth.get("summary", {}),
        synth.get("pairs_vs_bm25", []),
        f"Synthetic corpus (n_obs={_n_obs(synth.get('summary'))})",
        ylim=(0.70, 1.10),
        label_fmt="{:.3f}",
    )
    _draw_panel(
        axes[1],
        gh.get("summary", {}),
        gh.get("pairs_vs_bm25", []),
        f"GitHub Discussions corpus (n_obs={_n_obs(gh.get('summary'))})",
        ylim=(0.965, 0.992),
        label_fmt="{:.4f}",
    )

    fig.tight_layout()
    out_dir = Path("data/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig14_linucb_comparison.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / "fig14_linucb_comparison.png", bbox_inches="tight", dpi=300
    )
    plt.close()
    print("fig14 done")


if __name__ == "__main__":
    main()
