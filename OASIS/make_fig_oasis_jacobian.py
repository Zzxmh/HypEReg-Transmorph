"""
Generate Figure: IXI→OASIS Jacobian comparison for HypEReg-TM, TransMorph, MIDIR.

Layout mirrors Fig. 4 in the manuscript:
  Top row   : det(Jφ) heatmap on axial central slice — one panel per model + shared colorbar
  Bottom row: log(det Jφ) histogram overlay — all three models in one wide panel

Run from repository root:
    python OASIS/make_fig_oasis_jacobian.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.ndimage import zoom

# ── repo paths ────────────────────────────────────────────────────────────────
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SUBMISSION = os.path.join(REPO, "OASIS", "data", "Submit", "submission")
OUT_DIR = os.path.join(REPO, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── models to compare ─────────────────────────────────────────────────────────
MODELS = [
    ("transmorph_her_zs_oasis",  "HypEReg-TransMorph\n(IXI→OASIS zero-shot)"),
    ("transmorph_zs_oasis",       "TransMorph\n(IXI→OASIS zero-shot)"),
    ("midir_oasis",               "MIDIR\n(IXI→OASIS zero-shot)"),
]

# representative case (first test pair)
CASE = "0438_0439"

# ── Jacobian computation ───────────────────────────────────────────────────────
def load_flow_full(model_id: str, case: str) -> np.ndarray:
    """Load half-res .npz, zoom ×2 → full (3, D, H, W) float32."""
    path = os.path.join(SUBMISSION, model_id, "task_03", f"disp_{case}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing: {path}")
    flow = np.load(path)["arr_0"].astype(np.float32)   # (3, 80, 96, 112)
    flow_full = np.array([zoom(flow[i], 2.0, order=2) for i in range(3)])
    return flow_full   # (3, 160, 192, 224)


def jacobian_det(flow: np.ndarray) -> np.ndarray:
    """Forward-difference Jacobian determinant on interior voxels → (D-1,H-1,W-1)."""
    dx = flow[:, 1:, :-1, :-1] - flow[:, :-1, :-1, :-1]
    dy = flow[:, :-1, 1:, :-1] - flow[:, :-1, :-1, :-1]
    dz = flow[:, :-1, :-1, 1:] - flow[:, :-1, :-1, :-1]
    F11, F21, F31 = dx[0] + 1, dx[1],     dx[2]
    F12, F22, F32 = dy[0],     dy[1] + 1, dy[2]
    F13, F23, F33 = dz[0],     dz[1],     dz[2] + 1
    J = (F11 * (F22 * F33 - F23 * F32)
         - F12 * (F21 * F33 - F23 * F31)
         + F13 * (F21 * F32 - F22 * F31))
    return J   # (D-1, H-1, W-1)


# ── figure ────────────────────────────────────────────────────────────────────
def make_figure() -> None:
    n_models = len(MODELS)

    # load flows and compute Jacobians
    jacs: list[np.ndarray] = []
    labels: list[str] = []
    for mid, label in MODELS:
        try:
            flow = load_flow_full(mid, CASE)
        except FileNotFoundError as e:
            print(f"[WARN] {e} — skipping")
            continue
        J = jacobian_det(flow)
        jacs.append(J)
        labels.append(label)

    if not jacs:
        print("No displacement fields found; aborting.")
        return

    n = len(jacs)
    D = jacs[0].shape[0]
    axial_slice = D // 2  # central axial slice

    # ── layout: (n+1) top columns (maps + colorbar), 1 bottom spanning row ──
    fig = plt.figure(figsize=(4 * n + 1.4, 8.5), dpi=200)
    gs = gridspec.GridSpec(
        2, n + 1,
        height_ratios=[1.05, 0.85],
        width_ratios=[1.0] * n + [0.08],
        hspace=0.10, wspace=0.04,
    )

    VMIN, VMAX = 0.5, 1.5   # matches Fig. 4 colour range
    CMAP = "coolwarm"

    ax_maps = []
    for i, (J, lab) in enumerate(zip(jacs, labels)):
        ax = fig.add_subplot(gs[0, i])
        ax_maps.append(ax)
        jslice = J[axial_slice]   # (H-1, W-1)
        im = ax.imshow(jslice, cmap=CMAP, vmin=VMIN, vmax=VMAX,
                       origin="lower", interpolation="nearest")
        ax.set_title(lab, fontsize=10, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])

        # annotate non-jec & SDlogJ in bottom-left corner
        non_jec = float((J <= 0).mean()) * 100
        logJ = np.log(np.clip(J, 1e-6, None))
        sdlogj = float(logJ.std())
        ax.text(0.02, 0.02,
                f"non-jec: {non_jec:.4f}%\nSDlogJ: {sdlogj:.4f}",
                transform=ax.transAxes, fontsize=7.5,
                verticalalignment="bottom", color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55))

    # shared colorbar
    ax_cb = fig.add_subplot(gs[0, n])
    plt.colorbar(im, cax=ax_cb, label=r"$\det(J_\phi)$")
    ax_cb.tick_params(labelsize=8)

    # ── bottom row: log-J histogram overlay ──────────────────────────────────
    ax_hist = fig.add_subplot(gs[1, :n])
    COLORS = ["#c0392b", "#2980b9", "#27ae60"]   # red, blue, green
    for J, lab, col in zip(jacs, labels, COLORS):
        logJ = np.log(np.clip(J.ravel(), 1e-6, None))
        ax_hist.hist(logJ, bins=120, density=True, alpha=0.55,
                     color=col, label=lab.replace("\n", " "), linewidth=0)
    ax_hist.axvline(0, color="k", linestyle="--", linewidth=1.0, label="log J = 0")
    ax_hist.set_xlabel(r"$\log\,\det(J_\phi)$", fontsize=11)
    ax_hist.set_ylabel("Density", fontsize=11)
    ax_hist.set_title("Log-Jacobian distribution (axial slice / full volume)",
                      fontsize=10)
    ax_hist.legend(fontsize=8.5, loc="upper right")
    ax_hist.set_xlim(-1.2, 1.2)

    # shared colorbar axis placeholder (keep alignment)
    ax_cb2 = fig.add_subplot(gs[1, n])
    ax_cb2.axis("off")

    fig.suptitle(
        f"OASIS zero-shot Jacobian analysis — case {CASE}",
        fontsize=12, y=1.01,
    )

    out = os.path.join(OUT_DIR, "fig_oasis_jacobian.pdf")
    fig.savefig(out, bbox_inches="tight")
    png_out = out.replace(".pdf", ".png")
    fig.savefig(png_out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Saved: {png_out}")


if __name__ == "__main__":
    sys.path.insert(0, REPO)
    make_figure()
