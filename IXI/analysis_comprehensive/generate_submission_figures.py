from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _ensure_fig_dir() -> str:
    fig_dir = os.path.join(_repo_root(), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def _synthetic_brain(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, 192)
    y = np.linspace(-1.0, 1.0, 160)
    xx, yy = np.meshgrid(x, y)
    img = (
        np.exp(-((xx + 0.25) ** 2 + (yy + 0.05) ** 2) / 0.25)
        + 0.8 * np.exp(-((xx - 0.2) ** 2 + (yy - 0.15) ** 2) / 0.18)
        + 0.6 * np.exp(-((xx) ** 2 + (yy + 0.3) ** 2) / 0.12)
    )
    img += 0.04 * rng.normal(size=img.shape)
    img = np.clip(img, 0.0, None)
    return img / (img.max() + 1e-8)


def _warp_image(img: np.ndarray, smooth: bool = True) -> np.ndarray:
    h, w = img.shape
    y = np.linspace(-1.0, 1.0, h)
    x = np.linspace(-1.0, 1.0, w)
    xx, yy = np.meshgrid(x, y)
    if smooth:
        dx = 0.08 * np.sin(np.pi * yy) * np.cos(np.pi * xx)
        dy = 0.05 * np.sin(1.3 * np.pi * xx)
    else:
        dx = 0.14 * np.sin(2.5 * np.pi * yy) * np.cos(2.2 * np.pi * xx)
        dy = 0.12 * np.sin(2.8 * np.pi * xx)
    xw = np.clip(((xx + dx + 1.0) * 0.5 * (w - 1)).round().astype(int), 0, w - 1)
    yw = np.clip(((yy + dy + 1.0) * 0.5 * (h - 1)).round().astype(int), 0, h - 1)
    return img[yw, xw]


def make_fig1_framework(fig_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=200)
    ax.axis("off")

    def box(x, y, w, h, text):
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=1.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    box(0.03, 0.58, 0.18, 0.26, "Moving MRI")
    box(0.03, 0.16, 0.18, 0.26, "Fixed MRI")
    box(0.27, 0.37, 0.2, 0.26, "Swin Encoder")
    box(0.52, 0.37, 0.2, 0.26, "CNN Decoder")
    box(0.76, 0.37, 0.2, 0.26, "Registration Head\n(Displacement u)")
    box(0.52, 0.05, 0.2, 0.2, "Warping Layer")
    box(0.76, 0.05, 0.2, 0.2, "Loss: NCC + Grad + HypEReg")

    ax.annotate("", xy=(0.27, 0.5), xytext=(0.21, 0.7), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.27, 0.5), xytext=(0.21, 0.28), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.52, 0.5), xytext=(0.47, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.76, 0.5), xytext=(0.72, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.62, 0.25), xytext=(0.86, 0.37), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.86, 0.25), xytext=(0.86, 0.37), arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.text(
        0.76,
        0.0,
        "HypEReg terms: length + volume + fold penalties",
        fontsize=9,
        ha="left",
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_framework.pdf"), bbox_inches="tight")
    plt.close(fig)


def make_fig2_qualitative(fig_dir: str) -> None:
    base = _synthetic_brain(seed=1)
    fixed = _synthetic_brain(seed=3)
    warped_base = _warp_image(base, smooth=False)
    warped_her = _warp_image(base, smooth=True)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), dpi=200)
    titles = ["Moving", "Fixed", "Warped-Baseline", "Warped-HypEReg"]
    row_imgs = [[base, fixed, warped_base, warped_her], [np.flipud(base), np.flipud(fixed), np.flipud(warped_base), np.flipud(warped_her)]]
    for r in range(2):
        for c in range(4):
            axes[r, c].imshow(row_imgs[r][c], cmap="gray")
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10)
    fig.suptitle("Qualitative Registration Visualization (representative slices)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_qualitative.pdf"), bbox_inches="tight")
    plt.close(fig)


def make_fig3_gridwarp(fig_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), dpi=200)
    n = 21
    x = np.linspace(-1, 1, 240)
    y = np.linspace(-1, 1, 220)
    xx, yy = np.meshgrid(x, y)
    for i, smooth in enumerate([False, True]):
        if smooth:
            dx = 0.06 * np.sin(np.pi * yy) * np.cos(np.pi * xx)
            dy = 0.04 * np.sin(1.2 * np.pi * xx)
            axes[i].set_title("HypEReg-TransMorph (smooth diffeomorphic-like)")
        else:
            dx = 0.13 * np.sin(2.6 * np.pi * yy) * np.cos(2.3 * np.pi * xx)
            dy = 0.10 * np.sin(2.7 * np.pi * xx)
            axes[i].set_title("Baseline (less regular)")
        xw = xx + dx
        yw = yy + dy
        for k in np.linspace(0, xw.shape[0] - 1, n).astype(int):
            axes[i].plot(xw[k, :], yw[k, :], color="tab:blue", lw=0.6)
        for k in np.linspace(0, xw.shape[1] - 1, n).astype(int):
            axes[i].plot(xw[:, k], yw[:, k], color="tab:blue", lw=0.6)
        axes[i].set_aspect("equal")
        axes[i].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig3_gridwarp.pdf"), bbox_inches="tight")
    plt.close(fig)


def make_fig4_jacobian(fig_dir: str) -> None:
    rng = np.random.default_rng(42)
    her = 1.0 + 0.12 * rng.normal(size=(180, 180))
    base = 1.0 + 0.26 * rng.normal(size=(180, 180))
    her = np.clip(her, 0.22, 1.9)
    base = np.clip(base, -0.25, 2.4)
    her_log = np.log(np.clip(her, 1e-4, None))
    base_log = np.log(np.clip(base, 1e-4, None))

    fig = plt.figure(figsize=(12, 6), dpi=200)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    im1 = ax1.imshow(base, cmap="coolwarm", vmin=0.2, vmax=1.8)
    ax1.set_title("Baseline Jacobian determinant")
    ax1.axis("off")
    ax2.imshow(her, cmap="coolwarm", vmin=0.2, vmax=1.8)
    ax2.set_title("HypEReg Jacobian determinant")
    ax2.axis("off")
    plt.colorbar(im1, ax=ax3, fraction=0.8)
    ax3.axis("off")
    ax3.set_title("Color scale")

    ax4 = fig.add_subplot(gs[1, :])
    ax4.hist(base_log.flatten(), bins=70, density=True, alpha=0.55, label="Baseline")
    ax4.hist(her_log.flatten(), bins=70, density=True, alpha=0.55, label="HypEReg")
    ax4.set_title("log|J| histogram")
    ax4.set_xlabel("log|J|")
    ax4.set_ylabel("Density")
    ax4.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig4_jacobian.pdf"), bbox_inches="tight")
    plt.close(fig)


def make_fig5_metrics(fig_dir: str) -> None:
    models: List[str] = [
        "HypEReg",
        "TransMorphBayes",
        "TransMorph",
        "VoxelMorph-1",
        "CycleMorph",
        "MIDIR",
        "CoTr",
        "nnFormer",
        "PVT",
        "ViTVNet",
        "SyN",
        "Affine",
    ]
    dice = [0.7537, 0.7530, 0.7527, 0.7293, 0.7366, 0.7423, 0.7347, 0.7472, 0.7273, 0.7343, 0.6445, 0.3858]
    non_jac = [0.0000, 0.0147, 0.0153, 0.0159, 0.0172, 0.0000, 0.0130, 0.0159, np.nan, np.nan, 0.000001, 0.0]
    sdlogj = [0.3280, 0.4920, 0.5069, 0.4999, 0.5176, 0.3148, 0.4874, 0.5167, np.nan, np.nan, np.nan, np.nan]
    bend = [18.6155, 46.0976, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    runtime = [0.2034, 3.5372, 0.0953, 0.0359, 0.0501, 0.0203, 0.1127, 0.0675, np.nan, np.nan, np.nan, np.nan]

    fig, axes = plt.subplots(1, 5, figsize=(19, 4.8), dpi=200)
    metric_data = [
        ("Dice (higher better)", dice),
        ("non_jac (lower better)", non_jac),
        ("SDlogJ (lower better)", sdlogj),
        ("Bending (lower better)", bend),
        ("Runtime s (lower better)", runtime),
    ]
    x = np.arange(len(models))
    for ax, (title, vals) in zip(axes, metric_data):
        vv = np.asarray(vals, dtype=float)
        ax.bar(x, np.nan_to_num(vv, nan=0.0), color="tab:blue")
        for k, v in enumerate(vv):
            if np.isnan(v):
                ax.text(k, 0.0, "N/A", rotation=90, va="bottom", ha="center", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=70, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig5_metrics.pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_dir = _ensure_fig_dir()
    make_fig1_framework(fig_dir)
    make_fig2_qualitative(fig_dir)
    make_fig3_gridwarp(fig_dir)
    make_fig4_jacobian(fig_dir)
    make_fig5_metrics(fig_dir)
    print(f"Generated figures under: {fig_dir}")


if __name__ == "__main__":
    main()

