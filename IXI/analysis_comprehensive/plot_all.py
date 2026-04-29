"""
Cross-model comparison figures and per-VOI boxplots.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from typing import List, Optional, Sequence

import numpy as np

if "MPLBACKEND" not in os.environ:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import MODEL_REGISTRY, ModelSpec, TRANSMORPH_DIR

if TRANSMORPH_DIR not in sys.path:
    sys.path.insert(0, TRANSMORPH_DIR)
from voi_definitions import VOI_LBLS  # noqa: E402


def _read_metric_matrix(csv_path: str) -> np.ndarray:
    if not os.path.isfile(csv_path):
        return np.zeros((0, len(VOI_LBLS)))
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return np.zeros((0, len(VOI_LBLS)))
    out: List[List[float]] = []
    for r in rows[1:]:
        out.append(
            [float(c) if c and c.lower() != "nan" else float("nan") for c in r[1:]]
        )
    return np.asarray(out, dtype=np.float64) if out else np.zeros((0, len(VOI_LBLS)))


def _boxplot_voi(
    out_root: str,
    save_stem: str,
    file_name: str,
    ylabel: str,
    ylim: Optional[tuple] = None,
    specs: Optional[Sequence[ModelSpec]] = None,
) -> None:
    specs = list(specs) if specs is not None else list(MODEL_REGISTRY)
    names = [s.name for s in specs]
    n_m, n_v = len(names), len(VOI_LBLS)
    mats: List[np.ndarray] = [
        _read_metric_matrix(os.path.join(out_root, s, file_name)) for s in names
    ]
    lens = [M.shape[0] for M in mats if M.shape[0] > 0]
    m = min(lens) if lens else 0
    if m == 0:
        return
    mats = [M[:m, :].copy() for M in mats if M.shape[0] >= m]

    spacing = max(12.0, 1.2 * n_m + 2)
    centers = np.arange(n_v) * spacing
    offsets = np.linspace(-(n_m - 1) / 2.0 * 0.35, (n_m - 1) / 2.0 * 0.35, n_m)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i / max(n_m - 1, 1)) for i in range(n_m)]
    flierprops = {"marker": "o", "markersize": 1, "alpha": 0.3}
    fig, ax = plt.subplots(figsize=(max(10, n_v * 0.3), 7), dpi=150)
    for vi in range(n_v):
        for mi in range(n_m):
            d = mats[mi][:, vi]
            d = d[np.isfinite(d)]
            if d.size < 1:
                continue
            pos = float(centers[vi] + offsets[mi])
            bp = ax.boxplot(
                d,
                positions=[pos],
                widths=0.3,
                patch_artist=True,
                flierprops=flierprops,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[mi])
                patch.set_alpha(0.55)
    ax.set_xticks(centers)
    ax.set_xticklabels([str(x) for x in VOI_LBLS], rotation=90, fontsize=6)
    ax.set_ylabel(ylabel, fontsize=10)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.25)
    leg = [plt.Line2D([0], [0], color=colors[i], lw=3, label=names[i]) for i in range(n_m)]
    ax.legend(handles=leg, loc="lower right", fontsize=7, ncol=2)
    fig.suptitle(f"{save_stem} per VOI (n={m} subjects)")
    os.makedirs(os.path.join(out_root, "figures"), exist_ok=True)
    p = os.path.join(out_root, "figures", f"boxplot_{save_stem}_per_voi.png")
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)


def plot_cross_model(
    out_root: str,
    specs: Optional[Sequence[ModelSpec]] = None,
) -> None:
    specs = list(specs) if specs is not None else list(MODEL_REGISTRY)
    names = [s.name for s in specs]
    n_m = len(names)
    figd = os.path.join(out_root, "figures")
    os.makedirs(figd, exist_ok=True)

    for save_stem, file_name, yl, ylim in [
        ("dice", "dice.csv", "DSC", (0, 1.02)),
        ("hd95", "hd95.csv", "HD95 (mm)", None),
        ("assd", "assd.csv", "ASSD (mm)", None),
        ("nsd1mm", "nsd1mm.csv", "NSD@1mm", (0, 1.02)),
    ]:
        _boxplot_voi(out_root, save_stem, file_name, yl, ylim=ylim, specs=specs)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for n in names:
        p = os.path.join(out_root, n, "jacobian_log_samples.npy")
        if not os.path.isfile(p):
            continue
        a = np.load(p)
        a = a[np.isfinite(a)]
        if a.size < 2:
            continue
        ax.hist(a, bins=100, alpha=0.35, label=n, density=True)
    ax.set_xlabel("log J  (J>0)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.set_title("Jacobian log (subsampled voxels, J>0)")
    fig.tight_layout()
    fig.savefig(os.path.join(figd, "hist_jacobian_overlay.png"), bbox_inches="tight")
    plt.close(fig)

    xs, nonj, labs = [], [], []
    for n in names:
        sj = os.path.join(out_root, n, "summary.json")
        if not os.path.isfile(sj):
            continue
        with open(sj) as f:
            s = json.load(f)
        v = s.get("per_subject", {}).get("non_jac_frac", {}).get("mean", 0.0) or 0.0
        xs.append(len(xs))
        nonj.append(float(v) * 100.0)
        labs.append(n)
    if xs:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        b = np.asarray(nonj, dtype=np.float64) + 1e-12
        ax.bar(list(range(len(labs))), b, color="steelblue", alpha=0.8)
        ax.set_xticks(list(range(len(labs))))
        ax.set_xticklabels(labs, rotation=25, ha="right")
        ax.set_ylabel("non_jac (%)")
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(figd, "bar_non_jac.png"), bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cmap = plt.get_cmap("tab10")
    for im, n in enumerate(names):
        p_d = _read_metric_matrix(os.path.join(out_root, n, "dice.csv"))
        p_jf = os.path.join(out_root, n, "jacobian.csv")
        if p_d.size == 0 or not os.path.isfile(p_jf):
            continue
        with open(p_jf) as f:
            jrows = list(csv.reader(f))
        njs = [float(r[1]) for r in jrows[1:]]
        L = min(p_d.shape[0], len(njs))
        xd = [float(np.nanmean(p_d[i, :])) for i in range(L)]
        yd = [njs[i] + 1e-12 for i in range(L)]
        col = cmap(im / max(n_m - 1, 1))
        ax.scatter(
            xd,
            np.log10(yd),
            s=10,
            alpha=0.45,
            color=col,
            label=n,
        )
    ax.set_xlabel("Per-subject mean Dice (30 VOIs)")
    ax.set_ylabel("log10(non_jac fraction)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(figd, "scatter_dice_vs_nonjec.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(max(6, 0.32 * len(VOI_LBLS)), max(3, 0.4 * n_m + 1)),
        dpi=150,
    )
    M = np.zeros((n_m, len(VOI_LBLS)))
    for i, n in enumerate(names):
        p = _read_metric_matrix(os.path.join(out_root, n, "dice.csv"))
        if p.shape[0] == 0:
            continue
        M[i, :] = np.nanmean(p, axis=0)
    imh = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(np.arange(n_m))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xticks(np.arange(len(VOI_LBLS)))
    ax.set_xticklabels([str(x) for x in VOI_LBLS], rotation=90, fontsize=5)
    ax.set_title("mean Dice: models × VOI")
    fig.colorbar(imh, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(figd, "heatmap_dice_per_voi.png"), bbox_inches="tight")
    plt.close(fig)
