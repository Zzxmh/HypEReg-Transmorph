"""
Generate fig3_gridwarp.pdf as a 3x3 grid:
  rows = 3 representative IXI test subjects
  cols = HypEReg-TransMorph | TransMorph | MIDIR

Usage: python make_fig3_3x3.py [--device cuda|cpu]
"""
from __future__ import annotations
import argparse
import glob
import importlib
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
IXI_DIR = REPO_ROOT / "IXI"
IXI_DATA = REPO_ROOT / "IXI_data"

if str(IXI_DIR) not in sys.path:
    sys.path.insert(0, str(IXI_DIR))

# ── selected subjects (indices into sorted Test/*.pkl) ──────────────────────
# p_67 = max HER/TM fold contrast, p_36 = high contrast + low SDlogJ, p_20 = median
SUBJECT_INDICES = [67, 36, 20]
SUBJECT_LABELS = ["Case A", "Case B", "Case C"]

MODELS = ["transmorph_her", "transmorph_original", "midir"]
MODEL_NAMES = ["HypEReg-TransMorph", "TransMorph", "MIDIR"]

GRID_STEP = 16  # lattice line spacing (voxels)


def load_atlas_and_subject(subject_pkl: Path):
    tm_data_dir = IXI_DIR / "TransMorph" / "data"
    if str(tm_data_dir) not in sys.path:
        sys.path.insert(0, str(tm_data_dir))
    from data_utils import pkload  # noqa: WPS433

    atlas_path = IXI_DATA / "atlas.pkl"
    if not atlas_path.exists():
        atlas_path = IXI_DATA / "altas.pkl"  # typo fallback
    x, _ = pkload(str(atlas_path))
    y, _ = pkload(str(subject_pkl))
    return x.astype(np.float32), y.astype(np.float32)


_BASE_SYSPATH = None  # snapshot of sys.path before any adapter touches it

def _reset_syspath():
    """Restore sys.path to its pre-adapter baseline and purge model/adapter modules."""
    global _BASE_SYSPATH
    if _BASE_SYSPATH is None:
        return
    # Remove any entries added by adapters
    to_remove = [p for p in sys.path if p not in _BASE_SYSPATH]
    for p in to_remove:
        sys.path.remove(p)
    # Purge all model/adapter module caches
    stale = [k for k in list(sys.modules.keys())
             if any(k == prefix or k.startswith(prefix + ".")
                    for prefix in ("models", "adapters", "TransMorph",
                                   "MIDIR", "data_utils", "losses"))]
    for k in stale:
        sys.modules.pop(k, None)


def build_and_forward(adapter_id: str, x_t: torch.Tensor, y_t: torch.Tensor, device: str):
    global _BASE_SYSPATH
    if _BASE_SYSPATH is None:
        _BASE_SYSPATH = list(sys.path)
    _reset_syspath()
    ad = importlib.import_module(f"adapters.{adapter_id}")
    model, _cfg = ad.build_model(device=device)
    model.eval()
    with torch.no_grad():
        _, flow = ad.forward(model, x_t, y_t)
    flow_np = flow.detach().cpu().numpy()[0]  # (3, D, H, W)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return flow_np


def _to_float01(vol: np.ndarray) -> np.ndarray:
    v = vol.astype(np.float32)
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    if hi <= lo:
        return np.clip(v, 0.0, 1.0)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def _jacobian_nonjec(flow: np.ndarray) -> float:
    """Quick non-positive Jacobian fraction from a (3,D,H,W) flow array."""
    # finite differences on interior
    u = flow  # (3, D, H, W)
    du_dx = u[0, 1:-1, 1:-1, 2:] - u[0, 1:-1, 1:-1, :-2]
    du_dy = u[1, 1:-1, 2:, 1:-1] - u[1, 1:-1, :-2, 1:-1]
    du_dz = u[2, 2:, 1:-1, 1:-1] - u[2, :-2, 1:-1, 1:-1]
    # diagonal of Jacobian (simplified det approximation for visualization)
    # Full det: use central-diff Jacobian along interior
    u0 = u[0, 1:-1, 1:-1, 1:-1]
    u1 = u[1, 1:-1, 1:-1, 1:-1]
    u2 = u[2, 1:-1, 1:-1, 1:-1]
    # Forward differences (same as training code)
    d00 = u[0, 1:-1, 1:-1, 2:] - u[0, 1:-1, 1:-1, 1:-1] + 1  # dφ0/dx0 + 1
    d01 = u[0, 1:-1, 2:, 1:-1] - u[0, 1:-1, 1:-1, 1:-1]
    d02 = u[0, 2:, 1:-1, 1:-1] - u[0, 1:-1, 1:-1, 1:-1]
    d10 = u[1, 1:-1, 1:-1, 2:] - u[1, 1:-1, 1:-1, 1:-1]
    d11 = u[1, 1:-1, 2:, 1:-1] - u[1, 1:-1, 1:-1, 1:-1] + 1
    d12 = u[1, 2:, 1:-1, 1:-1] - u[1, 1:-1, 1:-1, 1:-1]
    d20 = u[2, 1:-1, 1:-1, 2:] - u[2, 1:-1, 1:-1, 1:-1]
    d21 = u[2, 1:-1, 2:, 1:-1] - u[2, 1:-1, 1:-1, 1:-1]
    d22 = u[2, 2:, 1:-1, 1:-1] - u[2, 1:-1, 1:-1, 1:-1] + 1
    det = (d00 * (d11 * d22 - d12 * d21)
           - d01 * (d10 * d22 - d12 * d20)
           + d02 * (d10 * d21 - d11 * d20))
    return float((det <= 0).mean())


GRID_COLOR = "#00CC44"   # vivid green
GRID_LW    = 1.0
GRID_ALPHA = 0.88


def add_nonjac_corner_label(
    ax,
    nj: float,
    color: str,
    fontsize_main: int = 20,
):
    """Lower-right inset for non-positive Jacobian ratio (mathtext: exponent as superscript)."""
    mantissa, _, exp_part = f"{nj:.2e}".partition("e")
    m = float(mantissa)
    exp_i = int(exp_part)
    tex = rf"$\mathrm{{J}}\!\leq\!0:\ {m:.2f}\times10^{{{exp_i}}}$"
    ax.text(
        0.98,
        0.02,
        tex,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=fontsize_main,
        color=color,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.60),
    )


def draw_grid(ax, bg: np.ndarray, flow: np.ndarray, step: int = GRID_STEP):
    """Draw deformed lattice on the central axial slice (dim-0 midpoint).

    Coordinate convention (matches the original working 1×3 figure):
      bg[z, :, :] → shape (H, W); np.rot90 → (W, H) displayed by imshow.
      imshow of shape (W, H): x-axis = H columns (0..H-1), y-axis = W rows (0..W-1).
      Grid x-coordinate = H-1 - ys_d   (H-axis, flipped for natural orientation)
      Grid y-coordinate = xs_d          (W-axis)
    """
    D, H, W = bg.shape
    z = D // 2
    bg2d = np.rot90(_to_float01(bg[z, :, :]))   # (W, H) → imshow: x∈[0,H-1], y∈[0,W-1]
    ax.imshow(bg2d, cmap="gray")

    # horizontal grid lines (constant y in original voxel space)
    for y0 in range(0, H, step):
        xs   = np.arange(W, dtype=np.float32)
        xs_d = xs + flow[2, z, y0, :]
        ys_d = np.full(W, y0, dtype=np.float32) + flow[1, z, y0, :]
        ax.plot(H - 1 - ys_d, xs_d,
                color=GRID_COLOR, lw=GRID_LW, alpha=GRID_ALPHA, solid_capstyle="round")

    # vertical grid lines (constant x in original voxel space)
    for x0 in range(0, W, step):
        ys   = np.arange(H, dtype=np.float32)
        xs_d = np.full(H, x0, dtype=np.float32) + flow[2, z, :, x0]
        ys_d = ys + flow[1, z, :, x0]
        ax.plot(H - 1 - ys_d, xs_d,
                color=GRID_COLOR, lw=GRID_LW, alpha=GRID_ALPHA, solid_capstyle="round")

    ax.axis("off")


def main(device: str):
    # resolve test PKL paths sorted the same way the eval scripts do
    test_pkls = sorted(glob.glob(str(IXI_DATA / "Test" / "*.pkl")))
    if not test_pkls:
        raise FileNotFoundError(f"No .pkl files found under {IXI_DATA}/Test/")
    print(f"Found {len(test_pkls)} test subjects.")

    selected_pkls = []
    for idx in SUBJECT_INDICES:
        if idx >= len(test_pkls):
            raise IndexError(f"Subject index {idx} out of range (total {len(test_pkls)})")
        selected_pkls.append(Path(test_pkls[idx]))
        print(f"  p_{idx:3d} -> {selected_pkls[-1].name}")

    # ── run inference for each (subject × model) ────────────────────────────
    flows = {}  # (subj_idx, model_id) -> np.ndarray (3, D, H, W)
    movings = {}
    for pi, pkl_path in zip(SUBJECT_INDICES, selected_pkls):
        print(f"\nLoading subject p_{pi}: {pkl_path.name}")
        moving, fixed = load_atlas_and_subject(pkl_path)
        movings[pi] = moving
        x_t = torch.from_numpy(moving[None, None]).float().to(device)
        y_t = torch.from_numpy(fixed[None, None]).float().to(device)
        for mid in MODELS:
            print(f"  Inference: {mid} ...", end=" ", flush=True)
            try:
                flow = build_and_forward(mid, x_t, y_t, device)
                flows[(pi, mid)] = flow
                print("OK")
            except Exception as e:
                print(f"FAILED: {e}")
                flows[(pi, mid)] = None

    # ── load per-subject non-pos-J from CSVs (consistent with evaluation) ───
    import csv

    def _load_nonjec_from_jacobian_csv(path):
        d = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                d[row["subject"]] = float(row["non_jac_frac"])
        return d

    def _load_nonjec_from_result_csv(path):
        d = {}
        with open(path) as f:
            reader = csv.reader(f)
            next(reader); next(reader)
            for row in reader:
                if len(row) >= 2:
                    try:
                        d[row[0]] = float(row[-1])
                    except ValueError:
                        pass
        return d

    her_jac_stats = _load_nonjec_from_jacobian_csv(
        REPO_ROOT / "IXI/Results/comprehensive/HER_dsc0743/jacobian.csv")
    tm_nonjec_stats = _load_nonjec_from_result_csv(
        REPO_ROOT / "IXI/Results/TransMorph_ncc_1_diffusion_1.csv")
    midir_nonjec_stats = _load_nonjec_from_result_csv(
        REPO_ROOT / "IXI/Results/MIDIR_ncc_1_diffusion_1.csv")

    nonjec_lookup = {
        "transmorph_her": her_jac_stats,
        "transmorph_original": tm_nonjec_stats,
        "midir": midir_nonjec_stats,
    }

    # ── build 3×3 figure ────────────────────────────────────────────────────
    nrows, ncols = len(SUBJECT_INDICES), len(MODELS)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.6 * ncols, 4.4 * nrows),
        dpi=300,
        gridspec_kw={"hspace": 0.06, "wspace": 0.04},
    )

    for ri, (pi, case_label) in enumerate(zip(SUBJECT_INDICES, SUBJECT_LABELS)):
        moving = movings[pi]
        subj_key = f"p_{pi}"
        for ci, (mid, mname) in enumerate(zip(MODELS, MODEL_NAMES)):
            ax = axes[ri, ci]
            flow = flows.get((pi, mid))
            if flow is not None:
                draw_grid(ax, moving, flow)
                # Annotation from pre-computed CSV (evaluation-consistent)
                nj = nonjec_lookup.get(mid, {}).get(subj_key)
                if nj is not None:
                    col = "red" if nj > 5e-3 else ("yellow" if nj > 3e-4 else "lime")
                    add_nonjac_corner_label(ax, nj, col)
            else:
                ax.set_facecolor("black")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="red")
                ax.axis("off")
            # Column titles on top row
            if ri == 0:
                ax.set_title(mname, fontsize=15, fontstyle="italic", pad=7)
            # Row label on left edge of leftmost panel (using text in axes coords)
            if ci == 0:
                ax.text(-0.10, 0.5, case_label,
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=13, fontweight="bold", rotation=90)

    fig.tight_layout(rect=[0.05, 0.0, 1.0, 1.0])
    out_path = Path(__file__).resolve().parent / "fig3_gridwarp.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # also save PNG for quick preview
    png_path = out_path.with_suffix(".png")
    doc = None
    try:
        import fitz
        doc = fitz.open(str(out_path))
        pg = doc[0]
        pix = pg.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        pix.save(str(png_path))
        doc.close()
        print(f"Preview: {png_path}")
    except Exception as e:
        print(f"Preview skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args.device)
