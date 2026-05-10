"""
Qualitative figure: T1 + ground-truth vs multi-atlas fused segmentations.

Grid layout: one row per OASIS test target, four columns
(Ground truth | HypEReg-TransMorph | TransMorph | MIDIR).

Requires local OASIS tensors (same layout as scripts/oasis_downstream.py), GPU
recommended. Caches fused label maps under OASIS/Eval_Results/downstream/fused_cache/.

Usage:
  python figures/render_downstream_qualitative.py --target-ids 440 444 448 --device cuda
  python figures/render_downstream_qualitative.py --from-cache-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import oasis_downstream as od  # noqa: E402

# OASIS 35-class labels highlighted in the figure (hippocampus, ventricles, thalamus)
HIGHLIGHT_RGBA: dict[int, tuple[float, float, float, float]] = {
    14: (0.86, 0.24, 0.20, 0.55),  # L hippocampus
    33: (0.55, 0.15, 0.45, 0.55),  # R hippocampus
    3: (0.12, 0.45, 0.85, 0.50),  # L ventricle
    22: (0.10, 0.65, 0.75, 0.50),  # R ventricle
    7: (0.18, 0.62, 0.22, 0.52),  # L thalamus
    26: (0.45, 0.75, 0.25, 0.52),  # R thalamus
}

# Column 0 = ground truth (no adapter); columns 1–3 = fused predictions
COLUMN_LABELS: list[str] = [
    "Ground truth",
    "HypEReg-TransMorph",
    "TransMorph",
    "MIDIR",
]
MODEL_ADAPTERS: list[str | None] = [
    None,
    "transmorph_her_zs_oasis",
    "transmorph_zs_oasis",
    "midir_oasis",
]


def _norm_slice(img2d: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(img2d, (2.0, 98.0))
    return np.clip((img2d - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)


def overlay_labels_on_mri(vol2d: np.ndarray, seg2d: np.ndarray) -> np.ndarray:
    g = _norm_slice(vol2d)
    rgb = np.stack([g, g, g], axis=-1)
    for lid, (r, gg, b, a) in HIGHLIGHT_RGBA.items():
        m = seg2d == lid
        if not np.any(m):
            continue
        base = rgb[m]
        rgb[m] = (1.0 - a) * base + a * np.array([r, gg, b], dtype=np.float32)
    return np.clip(rgb, 0.0, 1.0)


def pick_slice_index(seg: np.ndarray, axis: int) -> int:
    lids = list(HIGHLIGHT_RGBA.keys())
    mask = np.isin(seg, lids)
    if axis == 0:
        prof = mask.sum(axis=(1, 2))
    elif axis == 1:
        prof = mask.sum(axis=(0, 2))
    else:
        prof = mask.sum(axis=(0, 1))
    if prof.max() == 0:
        return int(seg.shape[axis] // 2)
    return int(np.argmax(prof))


def extract_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return vol[idx, :, :]
    if axis == 1:
        return vol[:, idx, :]
    return vol[:, :, idx]


def cache_path(cache_dir: Path, adapter_id: str, tid: int) -> Path:
    return cache_dir / f"fused_{adapter_id}_{tid:04d}.npz"


def load_or_compute_fused(
    adapter_id: str,
    atlases: dict[int, tuple[np.ndarray, np.ndarray]],
    tim: np.ndarray,
    tid: int,
    device: str,
    cache_dir: Path,
    from_cache_only: bool,
    force: bool,
) -> np.ndarray:
    cp = cache_path(cache_dir, adapter_id, tid)
    if cp.is_file() and not force:
        z = np.load(cp)
        return z["fused"].astype(np.int32)
    if from_cache_only:
        raise FileNotFoundError(f"Missing cache {cp}; run without --from-cache-only first.")
    fused = od.compute_fused_for_target(adapter_id, atlases, tim, device)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cp, fused=fused.astype(np.int16))
    return fused


def _load_atlases() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    atlases: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for aid in od.ATLAS_IDS:
        try:
            atlases[aid] = od.load_atlas(aid)
        except FileNotFoundError:
            pass
    if len(atlases) < 2:
        raise FileNotFoundError(
            f"Need atlas pickles under OASIS/data/All/ (found {len(atlases)}). "
            "See scripts/oasis_downstream.py data layout."
        )
    return atlases


def render_figure(
    target_ids: list[int],
    device: str,
    slice_axis: int,
    slice_index: int | None,
    cache_dir: Path,
    from_cache_only: bool,
    force: bool,
    out_dir: Path,
) -> tuple[Path, Path]:
    if len(target_ids) < 1:
        raise ValueError("Need at least one --target-id")

    atlases = _load_atlases()
    nrows = len(target_ids)
    ncols = len(COLUMN_LABELS)

    fig_w = 3.05 * ncols
    fig_h = 3.35 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=200, constrained_layout=True)
    axes = np.atleast_2d(axes)

    fused_cache: dict[tuple[str, int], np.ndarray] = {}

    for i, tid in enumerate(target_ids):
        tim, tgt_seg = od.load_target(tid)
        idx = slice_index if slice_index is not None else pick_slice_index(tgt_seg, slice_axis)
        img2d = extract_slice(tim, slice_axis, idx)
        gt2d = extract_slice(tgt_seg, slice_axis, idx)

        for j in range(ncols):
            adapter = MODEL_ADAPTERS[j]
            if adapter is None:
                seg2d = gt2d
            else:
                key = (adapter, tid)
                if key not in fused_cache:
                    fused_cache[key] = load_or_compute_fused(
                        adapter, atlases, tim, tid, device, cache_dir, from_cache_only, force
                    )
                seg2d = extract_slice(fused_cache[key], slice_axis, idx)

            rgb = overlay_labels_on_mri(img2d, seg2d)
            ax = axes[i, j]
            ax.imshow(rgb, origin="lower", interpolation="nearest")
            ax.axis("off")
            if i == 0:
                ax.set_title(COLUMN_LABELS[j], fontsize=9)
            if j == 0:
                ax.text(
                    -0.04,
                    0.5,
                    f"Target {tid}",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=9,
                    clip_on=False,
                )

    axis_name = ("sagittal", "coronal", "axial")[min(slice_axis, 2)]
    id_str = ", ".join(str(t) for t in target_ids)
    slice_note = f"slice {slice_index}" if slice_index is not None else f"ROI-informed {axis_name} slice per row"
    fig.suptitle(
        f"OASIS multi-atlas fusion (targets {id_str}); {slice_note}; "
        r"$N_{\mathrm{atlas}}=6$, IXI$\to$OASIS zero-shot",
        fontsize=11,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "fig6_downstream_segmentation.pdf"
    out_png = out_dir / "fig6_downstream_segmentation.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_pdf, out_png


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--target-ids",
        type=int,
        nargs="+",
        default=[440, 444, 448],
        help="OASIS Test_nii subject ids (one row each)",
    )
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--slice-axis", type=int, default=2, help="0=sagittal, 1=coronal, 2=axial")
    ap.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Fixed slice index for all rows (default: auto per target)",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "OASIS" / "Eval_Results" / "downstream" / "fused_cache",
    )
    ap.add_argument("--from-cache-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures")
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    render_figure(
        list(args.target_ids),
        device,
        args.slice_axis,
        args.slice_index,
        args.cache_dir,
        args.from_cache_only,
        args.force,
        args.out_dir,
    )
    print(f"Wrote fig6 ({len(args.target_ids)}x4 panels) to {args.out_dir}")


if __name__ == "__main__":
    main()
