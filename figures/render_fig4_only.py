"""Standalone re-render of fig4_jacobian (heatmaps + log-Jacobian histogram).

Re-runs inference for HypEReg-TransMorph, TransMorph, and MIDIR on a single
IXI test subject, then renders the figure with three panel titles in the
SAME font/style and a histogram legend that includes all three models.

Outputs both ``fig4_jacobian.pdf`` (for the manuscript build) and
``fig4_jacobian.png`` (for direct preview / sharing).
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
IXI_DIR = REPO_ROOT / "IXI"
IXI_DATA = REPO_ROOT / "IXI_data"
FIG_DIR = REPO_ROOT / "figures"
DEFAULT_SUBJECT = "subject_1.pkl"


DISPLAY_NAME = {
    "transmorph_her": "HypEReg-TransMorph",
    "transmorph_original": "TransMorph",
    "midir": "MIDIR",
}

# Stable, distinguishable colors for the histogram (also used for any
# per-model accent in the heatmap row if desired).
MODEL_COLOR = {
    "transmorph_her": "#d62728",   # red
    "transmorph_original": "#1f77b4",  # blue
    "midir": "#2ca02c",            # green
}


def _load_case(subject_name: str) -> Dict[str, np.ndarray]:
    tm_data_dir = IXI_DIR / "TransMorph" / "data"
    if str(tm_data_dir) not in sys.path:
        sys.path.insert(0, str(tm_data_dir))
    from data_utils import pkload  # noqa: WPS433

    atlas_path = IXI_DATA / "atlas.pkl"
    subj_path = IXI_DATA / "Test" / subject_name
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas not found: {atlas_path}")
    if not subj_path.exists():
        import glob
        all_subj = sorted(glob.glob(str(IXI_DATA / "Test" / "*.pkl")))
        if not all_subj:
            raise FileNotFoundError("No IXI test subjects found.")
        subj_path = Path(all_subj[0])

    x, _ = pkload(str(atlas_path))
    y, _ = pkload(str(subj_path))
    return {
        "moving": x.astype(np.float32),
        "fixed": y.astype(np.float32),
        "subject": subj_path.name,
    }


def _build_and_forward(adapter_id: str, x_t, y_t, device: str) -> Dict[str, np.ndarray]:
    if str(IXI_DIR) not in sys.path:
        sys.path.insert(0, str(IXI_DIR))
    ad = importlib.import_module(f"adapters.{adapter_id}")
    model, _ = ad.build_model(device=device)
    model.eval()
    with torch.no_grad():
        xw, flow = ad.forward(model, x_t, y_t)
    out = {
        "warped": xw.detach().cpu().numpy()[0, 0],
        "flow": flow.detach().cpu().numpy()[0],
    }
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def _compute_jacobians(model_outs: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if str(IXI_DIR) not in sys.path:
        sys.path.insert(0, str(IXI_DIR))
    from metrics_full import jacobian_determinant_vxm_np  # noqa: WPS433

    return {
        key: jacobian_determinant_vxm_np(out["flow"]).astype(np.float32)
        for key, out in model_outs.items()
    }


def render(jac_map: Dict[str, np.ndarray], out_pdf: Path, out_png: Path) -> None:
    order = [k for k in ("transmorph_her", "transmorph_original", "midir") if k in jac_map]
    n = len(order)

    fig = plt.figure(figsize=(12, 7), dpi=200)
    gs = fig.add_gridspec(
        2, n + 1,
        height_ratios=[1.0, 1.05],
        width_ratios=[1.0] * n + [0.05],
        wspace=0.10,
        hspace=0.30,
    )

    last_im = None
    for i, key in enumerate(order):
        ax = fig.add_subplot(gs[0, i])
        j = jac_map[key]
        z = j.shape[0] // 2
        hm = np.rot90(np.clip(j[z], 0.0, 2.0))
        last_im = ax.imshow(hm, cmap="coolwarm", vmin=0.5, vmax=1.5)
        ax.set_title(
            f"{DISPLAY_NAME[key]} Jacobian",
            fontsize=16,
            fontstyle="italic",
            fontweight="normal",
            family="DejaVu Sans",
            pad=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)

    cax = fig.add_subplot(gs[0, n])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(r"$\det(J_\phi)$", fontsize=14, fontstyle="italic", family="DejaVu Sans")

    # Histogram row spans all heatmap columns (excluding the colorbar slot).
    axh = fig.add_subplot(gs[1, :n])
    for key in order:
        j = jac_map[key]
        pos = j[j > 0]
        if pos.size == 0:
            continue
        axh.hist(
            np.log(pos),
            bins=80,
            range=(-5, 5),
            density=True,
            alpha=0.45,
            color=MODEL_COLOR[key],
            label=DISPLAY_NAME[key],
            edgecolor="none",
        )
    axh.set_xlabel(r"$\log\det(J_\phi)$", fontsize=15, fontstyle="italic", family="DejaVu Sans")
    axh.set_ylabel("Density", fontsize=15, fontstyle="italic", family="DejaVu Sans")
    axh.tick_params(axis="both", labelsize=11)
    axh.set_xlim(-5, 5)
    axh.grid(True, axis="y", linestyle="--", alpha=0.30)
    axh.set_axisbelow(True)
    leg = axh.legend(
        fontsize=12,
        prop={"family": "DejaVu Sans", "style": "italic", "size": 12},
        loc="upper right",
        frameon=True,
        framealpha=0.9,
    )
    leg.get_frame().set_edgecolor("#888888")

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.10, dpi=300)
    plt.close(fig)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    case = _load_case(DEFAULT_SUBJECT)
    moving, fixed = case["moving"], case["fixed"]
    print(f"[fig4] subject = {case['subject']}, device = {device}")

    x_t = torch.from_numpy(moving[None, None]).float().to(device)
    y_t = torch.from_numpy(fixed[None, None]).float().to(device)

    model_outs: Dict[str, Dict[str, np.ndarray]] = {}
    for mid in ("transmorph_her", "transmorph_original", "midir"):
        try:
            print(f"[fig4] running inference: {mid}", flush=True)
            model_outs[mid] = _build_and_forward(mid, x_t, y_t, device=device)
        except Exception as exc:
            print(f"[fig4] FAILED {mid}: {exc}", flush=True)
            raise

    jac_map = _compute_jacobians(model_outs)
    out_pdf = FIG_DIR / "fig4_jacobian.pdf"
    out_png = FIG_DIR / "fig4_jacobian.png"
    render(jac_map, out_pdf, out_png)
    print(f"[fig4] wrote {out_pdf}")
    print(f"[fig4] wrote {out_png}")


if __name__ == "__main__":
    main()
