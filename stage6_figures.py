"""
Stage 6: Publication-quality figures from outputs/results.csv.
Outputs saved to outputs/figures/ as .pdf and .png.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Georgia", "Times New Roman", "Times", "serif"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

CSV_PATH = Path("outputs/results.csv")
OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RATIOS_ORDERED = [0.9, 0.7, 0.5, 0.3, 0.1]
COMPRESSORS = ["abstractive", "llmlingua", "tfidf"]
DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]

COMP_LABELS = {
    "abstractive": "Abstractive (GPT-4o)",
    "llmlingua": "LLMLingua",
    "tfidf": "TF-IDF",
}
DS_LABELS = {
    "musique": "MuSiQue",
    "hotpotqa": "HotpotQA",
    "2wikimultihopqa": "2WikiMultihopQA",
}

COMP_COLORS = {
    "abstractive": "#2166ac",
    "llmlingua": "#d6604d",
    "tfidf": "#4dac26",
}
DS_COLORS = {
    "musique": "#7b2d8b",
    "hotpotqa": "#e08214",
    "2wikimultihopqa": "#1a9850",
}
DS_MARKERS = {"musique": "o", "hotpotqa": "s", "2wikimultihopqa": "^"}
COMP_MARKERS = {"abstractive": "D", "llmlingua": "o", "tfidf": "s"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: Path) -> dict:
    """Returns nested dict: data[compressor][dataset][ratio] = f1."""
    data: dict = defaultdict(lambda: defaultdict(dict))
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            c = row["compressor"]
            d = row["dataset"]
            r = float(row["ratio"])
            data[c][d][r] = float(row["f1"])
    return data


def mean_across_datasets(data: dict, compressor: str) -> list[float]:
    return [
        float(np.mean([data[compressor][ds][r] for ds in DATASETS]))
        for r in RATIOS_ORDERED
    ]


def mean_across_ratios(data: dict) -> dict[str, dict[str, float]]:
    """Returns means[compressor][ratio_str] = mean_f1."""
    out: dict = {}
    for c in COMPRESSORS:
        out[c] = {r: float(np.mean([data[c][ds][r] for ds in DATASETS]))
                  for r in RATIOS_ORDERED}
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _invert_x(ax: plt.Axes) -> None:
    ax.set_xlim(max(RATIOS_ORDERED) + 0.04, min(RATIOS_ORDERED) - 0.04)
    ax.set_xticks(RATIOS_ORDERED)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(p)
        print(f"Saved: {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — Compression curves per compressor (3 subplots)
# ---------------------------------------------------------------------------

def figure1(data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    fig.suptitle("Figure 1 — Compression Curves per Compressor", fontsize=12, y=1.02)

    for ax, comp in zip(axes, COMPRESSORS):
        for ds in DATASETS:
            f1s = [data[comp][ds][r] for r in RATIOS_ORDERED]
            ax.plot(RATIOS_ORDERED, f1s,
                    color=DS_COLORS[ds], marker=DS_MARKERS[ds],
                    linewidth=1.4, markersize=5, label=DS_LABELS[ds], alpha=0.85)

        # Mean line
        means = mean_across_datasets(data, comp)
        ax.plot(RATIOS_ORDERED, means,
                color="black", linewidth=2.2, linestyle="--",
                marker="D", markersize=6, label="Mean", zorder=5)

        # Threshold zone at ratio=0.3
        ax.axvline(0.3, color="#b2182b", linewidth=1.2, linestyle=":", alpha=0.8)
        ax.axvspan(0.3 - 0.02, 0.3 + 0.02, color="#b2182b", alpha=0.08)
        ax.text(0.3, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.02,
                "threshold", color="#b2182b", fontsize=7.5,
                ha="center", va="bottom", rotation=90,
                transform=ax.get_xaxis_transform())

        _invert_x(ax)
        _despine(ax)
        ax.set_xlabel("Compression ratio\n(← more compressed)")
        ax.set_title(COMP_LABELS[comp], pad=6)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    axes[0].set_ylabel("F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.06), fontsize=9)
    fig.tight_layout()
    _save(fig, "fig1_curves_per_compressor")


# ---------------------------------------------------------------------------
# Figure 2 — Heatmap: compressor × ratio, mean F1 across datasets
# ---------------------------------------------------------------------------

def figure2(data: dict) -> None:
    means = mean_across_ratios(data)
    matrix = np.array([
        [means[c][r] for r in RATIOS_ORDERED]
        for c in COMPRESSORS
    ])

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.suptitle("Figure 2 — Mean F1 Heatmap (Compressor × Ratio, averaged across datasets)",
                 fontsize=11, y=1.02)

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03, label="Mean F1")

    ax.set_xticks(range(len(RATIOS_ORDERED)))
    ax.set_xticklabels([f"{r:.1f}" for r in RATIOS_ORDERED])
    ax.set_yticks(range(len(COMPRESSORS)))
    ax.set_yticklabels([COMP_LABELS[c] for c in COMPRESSORS])
    ax.set_xlabel("Compression ratio (0.9 = least compressed → 0.1 = most)")
    ax.set_title("")

    for i, comp in enumerate(COMPRESSORS):
        for j, r in enumerate(RATIOS_ORDERED):
            val = matrix[i, j]
            text_color = "black" if 0.3 < val < 0.8 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9.5, color=text_color, fontweight="bold")

    _despine(ax)
    fig.tight_layout()
    _save(fig, "fig2_heatmap")


# ---------------------------------------------------------------------------
# Figure 3 — Compression curves per dataset (3 subplots)
# ---------------------------------------------------------------------------

def figure3(data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    fig.suptitle("Figure 3 — Compression Curves per Dataset", fontsize=12, y=1.02)

    for ax, ds in zip(axes, DATASETS):
        for comp in COMPRESSORS:
            f1s = [data[comp][ds][r] for r in RATIOS_ORDERED]
            ax.plot(RATIOS_ORDERED, f1s,
                    color=COMP_COLORS[comp], marker=COMP_MARKERS[comp],
                    linewidth=1.6, markersize=5.5,
                    label=COMP_LABELS[comp], alpha=0.9)

        _invert_x(ax)
        _despine(ax)
        ax.set_xlabel("Compression ratio\n(← more compressed)")
        ax.set_title(DS_LABELS[ds], pad=6)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    axes[0].set_ylabel("F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.06), fontsize=9)
    fig.tight_layout()
    _save(fig, "fig3_curves_per_dataset")


# ---------------------------------------------------------------------------
# Figure 4 — Abstractive flatness vs extractive threshold collapse
# ---------------------------------------------------------------------------

def figure4(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.suptitle(
        "Figure 4 — Abstractive Flatness vs. Extractive Threshold Collapse\n"
        "(mean F1 across all 3 datasets)",
        fontsize=11, y=1.03,
    )

    # Shaded threshold collapse zone: ratio 0.1–0.3
    ax.axvspan(0.1, 0.3, color="#b2182b", alpha=0.09, label="Threshold collapse zone (ratio 0.1–0.3)")
    ax.axvline(0.3, color="#b2182b", linewidth=1.0, linestyle=":", alpha=0.6)
    ax.axvline(0.1, color="#b2182b", linewidth=1.0, linestyle=":", alpha=0.6)

    for comp in COMPRESSORS:
        means = mean_across_datasets(data, comp)
        ax.plot(RATIOS_ORDERED, means,
                color=COMP_COLORS[comp], marker=COMP_MARKERS[comp],
                linewidth=2.2, markersize=7,
                label=COMP_LABELS[comp], zorder=4)

    # Annotate Δ range for abstractive
    abs_means = mean_across_datasets(data, "abstractive")
    abs_max = max(abs_means)
    abs_min = min(abs_means)
    delta = abs_max - abs_min
    mid_ratio = 0.5
    mid_f1 = abs_means[RATIOS_ORDERED.index(mid_ratio)]
    ax.annotate(
        f"Abstractive Δ = {delta:.3f}\n(near-flat across all ratios)",
        xy=(mid_ratio, mid_f1),
        xytext=(0.62, mid_f1 + 0.12),
        fontsize=8.5,
        color=COMP_COLORS["abstractive"],
        arrowprops=dict(arrowstyle="->", color=COMP_COLORS["abstractive"],
                        lw=1.2, connectionstyle="arc3,rad=0.2"),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COMP_COLORS["abstractive"],
                  alpha=0.85, lw=0.8),
    )

    # Annotate LLMLingua collapse
    ll_means = mean_across_datasets(data, "llmlingua")
    ll_01 = ll_means[RATIOS_ORDERED.index(0.1)]
    ll_09 = ll_means[RATIOS_ORDERED.index(0.9)]
    ax.annotate(
        f"LLMLingua: {ll_09:.2f}→{ll_01:.2f}\n({((ll_09-ll_01)/ll_09*100):.0f}% drop)",
        xy=(0.1, ll_01),
        xytext=(0.22, ll_01 - 0.12),
        fontsize=8.5,
        color=COMP_COLORS["llmlingua"],
        arrowprops=dict(arrowstyle="->", color=COMP_COLORS["llmlingua"], lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COMP_COLORS["llmlingua"],
                  alpha=0.85, lw=0.8),
    )

    _invert_x(ax)
    _despine(ax)
    ax.set_xlabel("Compression ratio (← more compressed)")
    ax.set_ylabel("Mean F1 (across MuSiQue, HotpotQA, 2WikiMultihopQA)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig4_abstractive_vs_extractive")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading {CSV_PATH} ...")
    data = load_data(CSV_PATH)
    print(f"Generating figures -> {OUT_DIR}/\n")

    figure1(data)
    figure2(data)
    figure3(data)
    figure4(data)

    print("\nDone. 4 figures × 2 formats = 8 files written.")


if __name__ == "__main__":
    main()
