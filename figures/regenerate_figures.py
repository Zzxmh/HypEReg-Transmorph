from __future__ import annotations

import argparse
import csv
import glob
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
IXI_DIR = REPO_ROOT / "IXI"
IXI_DATA = REPO_ROOT / "IXI_data"
DEFAULT_SUBJECT = "subject_1.pkl"
OUTSTRUCT = [
    "Brain-Stem",
    "Thalamus",
    "Cerebellum-Cortex",
    "Cerebral-White-Matter",
    "Cerebellum-White-Matter",
    "Putamen",
    "VentralDC",
    "Pallidum",
    "Caudate",
    "Lateral-Ventricle",
    "Hippocampus",
    "3rd-Ventricle",
    "4th-Ventricle",
    "Amygdala",
    "Cerebral-Cortex",
    "CSF",
    "choroid-plexus",
]
RESULTS_CSV_BY_MODEL = {
    "transmorph_her": "TransMorph_HER_IXI.csv",
    "transmorph_original": "TransMorph_ncc_1_diffusion_1.csv",
    "transmorphbayes": "TransMorphBayes_ncc_1_diffusion_1.csv",
    "voxelmorph_1": "Vxm_1_ncc_1_diffusion_1.csv",
    "cyclemorph": "CycleMorph.csv",
    "midir": "MIDIR_ncc_1_diffusion_1.csv",
    "cotr": "CoTr_ncc_1_diffusion_1.csv",
    "nnformer": "nnFormer_ncc_1_diffusion_1.csv",
    "pvt": "PVT_ncc_1_diffusion_1.csv",
    "syn": "ants_IXI.csv",
}


def _display_name(model_id: str) -> str:
    mid = model_id.strip().lower()
    if mid == "transmorph_her":
        return "HypEReg-TransMorph"
    if mid == "transmorph_original":
        return "TransMorph"
    if mid == "midir":
        return "MIDIR"
    if mid == "transmorphbayes":
        return "TransMorphBayes"
    if mid == "voxelmorph_1":
        return "VoxelMorph-1"
    if mid == "cyclemorph":
        return "CycleMorph"
    if mid == "cotr":
        return "CoTr"
    if mid == "nnformer":
        return "nnFormer"
    if mid == "pvt":
        return "PVT"
    if mid == "syn":
        return "SyN"
    return model_id.replace("_", "-")


def _to_float01(vol: np.ndarray) -> np.ndarray:
    v = vol.astype(np.float32)
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    if hi <= lo:
        return np.clip(v, 0.0, 1.0)
    v = (v - lo) / (hi - lo)
    return np.clip(v, 0.0, 1.0)


def _slice_2d(vol: np.ndarray, plane: str) -> np.ndarray:
    d, h, w = vol.shape
    if plane == "axial":
        arr = vol[d // 2, :, :]
    elif plane == "coronal":
        arr = vol[:, h // 2, :]
    elif plane == "sagittal":
        arr = vol[:, :, w // 2]
    else:
        raise ValueError(f"Unknown plane: {plane}")
    return np.rot90(arr)


def _build_and_forward(
    adapter_id: str,
    x_t: torch.Tensor,
    y_t: torch.Tensor,
    device: str,
) -> Dict[str, np.ndarray]:
    if str(IXI_DIR) not in sys.path:
        sys.path.insert(0, str(IXI_DIR))
    ad = importlib.import_module(f"adapters.{adapter_id}")
    model, _cfg = ad.build_model(device=device)
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
        all_subj = sorted(glob.glob(str(IXI_DATA / "Test" / "*.pkl")))
        if not all_subj:
            raise FileNotFoundError("No IXI test subjects found.")
        subj_path = Path(all_subj[0])

    x, _x_seg = pkload(str(atlas_path))
    y, _y_seg = pkload(str(subj_path))
    return {"moving": x.astype(np.float32), "fixed": y.astype(np.float32), "subject": subj_path.name}


def render_fig2(out_pdf: Path, moving: np.ndarray, fixed: np.ndarray, model_outs: Dict[str, Dict[str, np.ndarray]]):
    order = [k for k in ("transmorph_her", "transmorph_original", "midir") if k in model_outs]
    col_titles = ["Moving", "Fixed"] + [_display_name(k) for k in order]
    planes = ["axial", "coronal", "sagittal"]
    fig, axes = plt.subplots(len(planes), len(col_titles), figsize=(3.2 * len(col_titles), 3.0 * len(planes)), dpi=300)
    if len(planes) == 1:
        axes = np.asarray([axes])
    for r, plane in enumerate(planes):
        for c, name in enumerate(col_titles):
            ax = axes[r, c]
            ax.axis("off")
            if name == "Moving":
                img = _slice_2d(_to_float01(moving), plane)
            elif name == "Fixed":
                img = _slice_2d(_to_float01(fixed), plane)
            else:
                key = {
                    "HypEReg-TransMorph": "transmorph_her",
                    "TransMorph": "transmorph_original",
                }.get(name, name.replace("-", "_").lower())
                img = _slice_2d(_to_float01(model_outs[key]["warped"]), plane)
            ax.imshow(img, cmap="gray")
            if r == 0:
                ax.set_title(name, fontsize=22, fontweight="bold", pad=10)
            if c == 0:
                ax.text(
                    -0.12,
                    0.5,
                    plane,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    fontsize=20,
                    fontweight="bold",
                )
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def _draw_deformed_grid(ax, bg: np.ndarray, flow: np.ndarray, step: int = 16):
    z = flow.shape[1] // 2
    bg2 = bg[z, :, :]
    h, w = bg2.shape
    ax.imshow(np.rot90(_to_float01(bg2)), cmap="gray")
    # draw in non-rotated coordinates, then rotate by plotting transformed coords
    for y0 in range(0, h, step):
        xs = np.arange(w)
        ys = np.full_like(xs, y0, dtype=np.float32)
        xs_d = xs + flow[2, z, y0, :]
        ys_d = ys + flow[1, z, y0, :]
        ax.plot(h - 1 - ys_d, xs_d, color="cyan", linewidth=1.0, alpha=0.85)
    for x0 in range(0, w, step):
        ys = np.arange(h)
        xs = np.full_like(ys, x0, dtype=np.float32)
        xs_d = xs + flow[2, z, :, x0]
        ys_d = ys + flow[1, z, :, x0]
        ax.plot(h - 1 - ys_d, xs_d, color="cyan", linewidth=1.0, alpha=0.85)
    ax.axis("off")


def render_fig3(out_pdf: Path, moving: np.ndarray, model_outs: Dict[str, Dict[str, np.ndarray]]):
    order = [k for k in ("transmorph_her", "transmorph_original", "midir") if k in model_outs]
    fig, axes = plt.subplots(1, len(order), figsize=(4.5 * max(1, len(order)), 4.2), dpi=300)
    if len(order) == 1:
        axes = [axes]
    for ax, key in zip(axes, order):
        _draw_deformed_grid(ax, moving, model_outs[key]["flow"])
        ax.set_title(_display_name(key), fontsize=16, fontstyle="italic")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def render_fig4(out_pdf: Path, model_outs: Dict[str, Dict[str, np.ndarray]]):
    if str(IXI_DIR) not in sys.path:
        sys.path.insert(0, str(IXI_DIR))
    from metrics_full import jacobian_determinant_vxm_np  # noqa: WPS433

    order = [k for k in ("transmorph_her", "transmorph_original", "midir") if k in model_outs]
    jac_map = {}
    for key in order:
        j = jacobian_determinant_vxm_np(model_outs[key]["flow"]).astype(np.float32)
        jac_map[key] = j

    fig = plt.figure(figsize=(12, 6), dpi=300)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])
    for i, key in enumerate(order[:3]):
        ax = fig.add_subplot(gs[0, i])
        j = jac_map[key]
        z = j.shape[0] // 2
        hm = np.rot90(np.clip(j[z], 0.0, 2.0))
        im = ax.imshow(hm, cmap="coolwarm", vmin=0.5, vmax=1.5)
        ax.set_title(f"{_display_name(key)} Jacobian", fontsize=16, fontstyle="italic")
        ax.axis("off")
    cax = fig.add_axes([0.92, 0.58, 0.015, 0.25])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=14)

    axh = fig.add_subplot(gs[1, :])
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
            alpha=0.35,
            label=_display_name(key),
        )
    axh.set_xlabel("log det(J_phi)", fontsize=16, fontstyle="italic")
    axh.set_ylabel("Density", fontsize=16, fontstyle="italic")
    axh.tick_params(axis="both", labelsize=12)
    axh.set_xlim(-5, 5)
    axh.legend(fontsize=14, prop={"style": "italic"})
    axh.set_title("Jacobian log-det distribution", fontsize=16, fontstyle="italic")
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(out_pdf)
    plt.close(fig)


def _read_metric_rows() -> List[dict]:
    cands = [
        REPO_ROOT / "IXI" / "Results" / "comprehensive" / "model_summary.csv",
        REPO_ROOT / "IXI" / "Eval_Results" / "_stats" / "model_summary.csv",
    ]
    for p in cands:
        if p.exists():
            with p.open("r", encoding="utf-8", newline="") as f:
                return list(csv.DictReader(f))
    raise FileNotFoundError("No model_summary.csv found for fig5.")


def _mean_std(arr: List[float]) -> Tuple[float, float]:
    if not arr:
        return float("nan"), float("nan")
    v = np.asarray(arr, dtype=np.float64)
    return float(np.mean(v)), float(np.std(v, ddof=0))


def _grouped_dice_nonjac_from_results() -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for model, fname in RESULTS_CSV_BY_MODEL.items():
        csv_path = REPO_ROOT / "IXI" / "Results" / fname
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) < 3:
            continue
        header = rows[1]
        data_rows = rows[2:]
        idx_map = {st: [i for i, n in enumerate(header) if st in n] for st in OUTSTRUCT}
        dice_case = []
        non_j_case = []
        for r in data_rows:
            if len(r) < len(header):
                continue
            try:
                st_vals = []
                for st in OUTSTRUCT:
                    idxs = idx_map[st]
                    if not idxs:
                        continue
                    v = [float(r[k]) for k in idxs]
                    st_vals.append(float(np.mean(v)))
                if st_vals:
                    dice_case.append(float(np.mean(st_vals)))
                non_j_case.append(float(r[-1]))
            except Exception:
                continue
        if dice_case and non_j_case:
            d_mean, d_std = _mean_std(dice_case)
            j_mean, j_std = _mean_std(non_j_case)
            out[model] = {
                "dice_mean": d_mean,
                "dice_std": d_std,
                "non_j_mean": j_mean,
                "non_j_std": j_std,
                "n": float(len(dice_case)),
            }
    return out


def _summary_lookup() -> Dict[str, dict]:
    rows = _read_metric_rows()
    return {r["model"].strip().lower(): r for r in rows}


def _table_metric_lookup() -> Dict[str, dict]:
    p = REPO_ROOT / "IXI" / "Results" / "comprehensive" / "table_model_x_metric.csv"
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["model"].strip().lower(): r for r in rows}


def _parse_pm(s: str) -> Tuple[float, float]:
    if s is None:
        return float("nan"), float("nan")
    text = str(s).strip()
    if "±" in text:
        lhs, rhs = text.split("±", 1)
        return float(lhs.strip()), float(rhs.strip())
    return float(text), float("nan")


def render_fig5(out_pdf: Path):
    grouped = _grouped_dice_nonjac_from_results()
    summary_by_model = _summary_lookup()
    table_by_model = _table_metric_lookup()
    preferred = [
        "transmorph_her",
        "transmorph_original",
        "transmorphbayes",
        "voxelmorph_1",
        "cyclemorph",
        "midir",
        "cotr",
        "nnformer",
        "pvt",
        "syn",
    ]
    models = [m for m in preferred if m in grouped]
    if not models:
        raise RuntimeError("No expected models found in IXI/Results csv files")

    dice = np.asarray([grouped[m]["dice_mean"] for m in models], dtype=np.float32)
    non_j = np.asarray([grouped[m]["non_j_mean"] for m in models], dtype=np.float32)
    dice_std = np.asarray([grouped[m]["dice_std"] for m in models], dtype=np.float32)
    non_j_std = np.asarray([grouped[m]["non_j_std"] for m in models], dtype=np.float32)
    n_cases = np.asarray([max(grouped[m].get("n", 1.0), 1.0) for m in models], dtype=np.float32)

    sdlog_vals = []
    hd95_vals = []
    sdlog_std_vals = []
    hd95_std_vals = []
    for m in models:
        if m == "transmorph_her" and "her_dsc0743" in table_by_model:
            row = table_by_model["her_dsc0743"]
            s_mean, s_std = _parse_pm(row.get("sdlogJ", "nan"))
            h_mean, h_std = _parse_pm(row.get("hd95_mm", "nan"))
            sdlog_vals.append(s_mean)
            hd95_vals.append(h_mean)
            sdlog_std_vals.append(s_std)
            hd95_std_vals.append(h_std)
        elif m == "transmorphbayes" and "transmorphbayes" in table_by_model:
            row = table_by_model["transmorphbayes"]
            s_mean, s_std = _parse_pm(row.get("sdlogJ", "nan"))
            h_mean, h_std = _parse_pm(row.get("hd95_mm", "nan"))
            sdlog_vals.append(s_mean)
            hd95_vals.append(h_mean)
            sdlog_std_vals.append(s_std)
            hd95_std_vals.append(h_std)
        else:
            row = summary_by_model.get(m)
            sdlog_vals.append(float(row["SDlogJ_mean"]) if row else float("nan"))
            hd95_vals.append(float(row["HD95_mean_mean"]) if row else float("nan"))
            sdlog_std_vals.append(float(row["SDlogJ_std"]) if row else float("nan"))
            hd95_std_vals.append(float(row["HD95_mean_std"]) if row else float("nan"))
    sdlog = np.asarray(sdlog_vals, dtype=np.float32)
    hd95 = np.asarray(hd95_vals, dtype=np.float32)
    sdlog_std = np.asarray(sdlog_std_vals, dtype=np.float32)
    hd95_std = np.asarray(hd95_std_vals, dtype=np.float32)
    labels = [_display_name(m) for m in models]
    x = np.arange(len(models))
    her_idx = models.index("transmorph_her") if "transmorph_her" in models else -1
    colors = ["tab:red" if i == her_idx else "tab:gray" for i in range(len(models))]
    sem_eps = np.maximum(np.sqrt(np.maximum(n_cases, 1.0)), 1.0)
    dice_err = dice_std / sem_eps
    non_j_err = non_j_std / sem_eps
    sdlog_err = sdlog_std / sem_eps
    hd95_err = hd95_std / sem_eps

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    panels = [
        (dice, dice_err, "Dice (higher better)"),
        (non_j, non_j_err, "det(J_phi) <= 0 ratio (lower better)"),
        (sdlog, sdlog_err, "SDlogJ (lower better)"),
        (hd95, hd95_err, "HD95 (mm, lower better)"),
    ]
    for ax, (vals, errs, title) in zip(axes.ravel(), panels):
        ax.bar(
            x,
            vals,
            yerr=errs,
            color=colors,
            width=0.75,
            capsize=3.0,
            error_kw={"elinewidth": 1.0, "ecolor": "black"},
        )
        ax.set_title(title, fontsize=18, fontstyle="italic")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, rotation=30, ha="right")
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.suptitle("IXI quantitative overview", fontsize=22, fontstyle="italic")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def main(fig_dir: str, subject: str, models: List[str], device: str):
    root = Path(fig_dir)
    case = _load_case(subject)
    moving = case["moving"]
    fixed = case["fixed"]

    x_t = torch.from_numpy(moving[None, None]).float().to(device)
    y_t = torch.from_numpy(fixed[None, None]).float().to(device)

    model_outs: Dict[str, Dict[str, np.ndarray]] = {}
    for mid in models:
        try:
            print(f"[figures] running inference: {mid} on {case['subject']}", flush=True)
            model_outs[mid] = _build_and_forward(mid, x_t, y_t, device=device)
        except Exception as exc:
            print(f"[figures] skip {mid}: {exc}", flush=True)

    if not model_outs:
        raise RuntimeError("No model inference succeeded; cannot render figures.")

    render_fig2(root / "fig2_qualitative.pdf", moving, fixed, model_outs)
    render_fig3(root / "fig3_gridwarp.pdf", moving, model_outs)
    render_fig4(root / "fig4_jacobian.pdf", model_outs)
    render_fig5(root / "fig5_metrics.pdf")
    print(f"[figures] Real IXI figures generated in {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate manuscript figures from IXI inference/results.")
    parser.add_argument(
        "--fig-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory containing fig*.pdf outputs.",
    )
    parser.add_argument(
        "--subject",
        default=DEFAULT_SUBJECT,
        help="Subject file name under IXI_data/Test (e.g., subject_1.pkl).",
    )
    parser.add_argument(
        "--models",
        default="transmorph_her,transmorph_original,midir",
        help="Comma-separated adapter ids from IXI/adapters.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Inference device.",
    )
    args = parser.parse_args()
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    main(args.fig_dir, args.subject, model_ids, args.device)
