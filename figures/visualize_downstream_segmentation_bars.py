"""Bar-chart summary of Table tab:multiatlas (no OASIS volume data required)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_table_values() -> dict[str, dict[str, float]]:
    """Values copied from main manuscript Table 'tab:multiatlas'."""
    return {
        "HypEReg-TransMorph": {
            "Single Dice": 0.7795,
            "Fused Dice": 0.8271,
            "Delta Dice": 0.0477,
            "Hippocampus": 0.8654,
            "Ventricle": 0.9056,
            "Thalamus": 0.9227,
        },
        "TransMorph": {
            "Single Dice": 0.7712,
            "Fused Dice": 0.8201,
            "Delta Dice": 0.0489,
            "Hippocampus": 0.8492,
            "Ventricle": 0.9006,
            "Thalamus": 0.8979,
        },
        "TransMorphBayes": {
            "Single Dice": 0.7597,
            "Fused Dice": 0.8058,
            "Delta Dice": 0.0460,
            "Hippocampus": 0.8341,
            "Ventricle": 0.8975,
            "Thalamus": 0.8846,
        },
        "MIDIR": {
            "Single Dice": 0.7161,
            "Fused Dice": 0.7696,
            "Delta Dice": 0.0534,
            "Hippocampus": 0.8238,
            "Ventricle": 0.8719,
            "Thalamus": 0.8955,
        },
    }


def make_downstream_figure(output_dir: Path) -> tuple[Path, Path]:
    values = _load_table_values()
    models = list(values.keys())
    colors = ["#C62828", "#1E88E5", "#6D4C41", "#43A047"]

    single = np.array([values[m]["Single Dice"] for m in models])
    fused = np.array([values[m]["Fused Dice"] for m in models])
    delta = np.array([values[m]["Delta Dice"] for m in models])
    hippo = np.array([values[m]["Hippocampus"] for m in models])
    vent = np.array([values[m]["Ventricle"] for m in models])
    thal = np.array([values[m]["Thalamus"] for m in models])

    fig = plt.figure(figsize=(11.5, 4.6), dpi=200, constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.15], wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(models))
    w = 0.35
    ax0.bar(x - w / 2, single, width=w, label="Single Dice", color="#90CAF9", edgecolor="black", linewidth=0.6)
    ax0.bar(x + w / 2, fused, width=w, label="Fused Dice", color="#EF9A9A", edgecolor="black", linewidth=0.6)
    ax0.set_xticks(x)
    ax0.set_xticklabels(models, rotation=18, ha="right")
    ax0.set_ylim(0.70, 0.85)
    ax0.set_ylabel("Dice")
    ax0.set_title("Single vs. Fused Dice")
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend(frameon=False, fontsize=8, loc="upper left")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(x, delta, color=colors, edgecolor="black", linewidth=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=18, ha="right")
    ax1.set_ylim(0.043, 0.056)
    ax1.set_ylabel("Delta Dice")
    ax1.set_title("Fusion Gain")
    ax1.grid(axis="y", alpha=0.25)
    for i, v in enumerate(delta):
        ax1.text(i, v + 0.0002, f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    ax2 = fig.add_subplot(gs[0, 2])
    r = np.arange(3)
    width = 0.18
    ax2.bar(r - 1.5 * width, [hippo[0], vent[0], thal[0]], width=width, color=colors[0], label=models[0], edgecolor="black", linewidth=0.5)
    ax2.bar(r - 0.5 * width, [hippo[1], vent[1], thal[1]], width=width, color=colors[1], label=models[1], edgecolor="black", linewidth=0.5)
    ax2.bar(r + 0.5 * width, [hippo[2], vent[2], thal[2]], width=width, color=colors[2], label=models[2], edgecolor="black", linewidth=0.5)
    ax2.bar(r + 1.5 * width, [hippo[3], vent[3], thal[3]], width=width, color=colors[3], label=models[3], edgecolor="black", linewidth=0.5)
    ax2.set_xticks(r)
    ax2.set_xticklabels(["Hippocampus", "Ventricle", "Thalamus"])
    ax2.set_ylim(0.80, 0.93)
    ax2.set_ylabel("Fused Dice")
    ax2.set_title("ROI Fused Dice")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(frameon=False, fontsize=7, loc="lower right")

    fig.suptitle("Downstream OASIS Multi-Atlas Segmentation (n=20, N_atlas=6)", fontsize=12)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = output_dir / "fig6_downstream_segmentation.pdf"
    out_png = output_dir / "fig6_downstream_segmentation.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_pdf, out_png


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    pdf, png = make_downstream_figure(repo_root / "figures")
    print(f"Wrote: {pdf}\nWrote: {png}")
