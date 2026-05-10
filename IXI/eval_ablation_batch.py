#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch evaluation for ablation experiments under repo_root/experiments/.

Metrics:
- Dice: grouped strategy identical to IXI/analysis_trans.py (keyword contains)
- Jacobian: non_jec (fraction J<=0)
- SDJ: SDlogJ (std of log(J) over positive Jacobians)
- HD95: grouped mean 95th-percentile Hausdorff distance (mm)
- ASSD: grouped mean average symmetric surface distance (mm)

Outputs:
- per_experiment/per_case.csv
- per_experiment/summary.json
- global summary CSV at IXI/Eval_Results/ablation_batch/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from natsort import natsorted
from torch.utils.data import DataLoader
from torchvision import transforms

IXI_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(IXI_DIR)

# Required column in per_case.csv header to detect compatible format
_REQUIRED_COLS = ("HD95_mean", "ASSD_mean")


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(REPO_ROOT, p))


def _setup_transmorph_path() -> str:
    tm = os.path.join(IXI_DIR, "TransMorph")
    if tm not in sys.path:
        sys.path.insert(0, tm)
    return tm


def _resolve_atlas(ixi_root: str) -> str:
    for name in ("atlas.pkl", "altas.pkl"):
        p = os.path.join(ixi_root, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"No atlas.pkl (or altas.pkl) found in {ixi_root}")


# from IXI/analysis_trans.py
OUTSTRUCT: List[str] = [
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


def _label_names_0_45() -> List[str]:
    _setup_transmorph_path()
    import utils as tm_utils  # noqa: WPS433

    # process_label() expects label_info.txt under CWD (IXI/TransMorph/)
    tm_dir = os.path.join(IXI_DIR, "TransMorph")
    cwd = os.getcwd()
    try:
        os.chdir(tm_dir)
        d = tm_utils.process_label()
    finally:
        os.chdir(cwd)
    return [str(d[i]) for i in range(46)]


def _build_group_indices(label_names: Sequence[str]) -> Dict[str, List[int]]:
    """Mimic IXI/analysis_trans.py: pick label indices whose name contains group keyword."""
    out: Dict[str, List[int]] = {}
    for g in OUTSTRUCT:
        idx = [i for i, name in enumerate(label_names) if g in str(name)]
        out[g] = idx if idx else []
    return out


def grouped_dice_from_per_label(
    dice_46: np.ndarray, group_map: Dict[str, List[int]]
) -> Tuple[np.ndarray, float]:
    """
    dice_46: (46,) float.
    Returns (group_values (len(OUTSTRUCT),), mean_over_groups).
    """
    gvals = np.full(len(OUTSTRUCT), np.nan, dtype=np.float64)
    for gi, g in enumerate(OUTSTRUCT):
        idx = group_map.get(g, [])
        if not idx:
            continue
        v = dice_46[np.asarray(idx, dtype=int)]
        gvals[gi] = float(np.nanmean(v)) if v.size else float("nan")
    return gvals, float(np.nanmean(gvals))


def _surface_pts(mask: np.ndarray, spacing: float) -> Optional[np.ndarray]:
    """
    Return (N,3) float32 physical-space coords of surface voxels, or None if empty.
    Surface = mask & ~minimum_filter(mask, 3) (equivalent to mask minus eroded mask).
    Uses minimum_filter (faster than binary_erosion for large volumes).
    """
    from scipy.ndimage import minimum_filter  # noqa: WPS433

    m = mask.astype(np.uint8, copy=False)
    if not m.any():
        return None
    ero = minimum_filter(m, size=3).view(np.uint8)
    surf = (m & ~ero).astype(bool)
    if not surf.any():
        # Degenerate case: structure is only 1 voxel thick; use the full mask
        surf = mask.astype(bool)
    coords = np.argwhere(surf).astype(np.float32) * spacing
    return coords


def _hd95_assd_kdtree(
    pred_g: np.ndarray, true_g: np.ndarray, spacing: float = 1.0
) -> Tuple[float, float]:
    """
    Fast HD95 and ASSD via cKDTree on surface point clouds.
    Both pred_g and true_g are boolean (D,H,W) masks.
    """
    from scipy.spatial import cKDTree  # noqa: WPS433

    pc = _surface_pts(pred_g, spacing)
    tc = _surface_pts(true_g, spacing)
    if pc is None or tc is None:
        return float("nan"), float("nan")

    tree_t = cKDTree(tc)
    d_p2t, _ = tree_t.query(pc, workers=-1)
    tree_p = cKDTree(pc)
    d_t2p, _ = tree_p.query(tc, workers=-1)

    hd95 = float(max(np.percentile(d_p2t, 95), np.percentile(d_t2p, 95)))
    assd = float(0.5 * (d_p2t.mean() + d_t2p.mean()))
    return hd95, assd


def grouped_hd95_assd(
    pred: np.ndarray,
    true: np.ndarray,
    group_map: Dict[str, List[int]],
    spacing: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute grouped mean HD95 and ASSD using the same 17-group aggregation as Dice.
    For each group the binary mask is the union of all constituent label indices.
    Uses cKDTree surface-distance (fast) instead of full-volume distance transforms.
    """
    hds: List[float] = []
    assds: List[float] = []
    for g in OUTSTRUCT:
        idx = group_map.get(g, [])
        if not idx:
            continue
        idx_arr = np.asarray(idx, dtype=np.int32)
        pred_g = np.isin(pred, idx_arr)
        true_g = np.isin(true, idx_arr)
        if not true_g.any() and not pred_g.any():
            continue
        h, a = _hd95_assd_kdtree(pred_g, true_g, spacing=spacing)
        if np.isfinite(h):
            hds.append(h)
        if np.isfinite(a):
            assds.append(a)
    return (
        float(np.nanmean(hds)) if hds else float("nan"),
        float(np.nanmean(assds)) if assds else float("nan"),
    )


def _parse_dsc_from_filename(fn: str) -> float:
    m = re.search(r"dsc([0-9]+\.[0-9]+)", fn)
    if not m:
        return float("-inf")
    try:
        return float(m.group(1))
    except Exception:
        return float("-inf")


def _select_ckpt(ckpts: Sequence[str], strategy: str) -> str:
    if not ckpts:
        raise FileNotFoundError("No checkpoint candidates.")
    if strategy == "latest":
        return natsorted(list(ckpts))[-1]
    if strategy == "best_dsc":
        return max(list(ckpts), key=lambda p: _parse_dsc_from_filename(os.path.basename(p)))
    raise ValueError(f"Unknown ckpt strategy: {strategy}")


def _find_experiments(exp_root: str, prefix: str) -> List[str]:
    if not os.path.isdir(exp_root):
        return []
    out = []
    for name in os.listdir(exp_root):
        p = os.path.join(exp_root, name)
        if os.path.isdir(p) and name.startswith(prefix):
            out.append(p)
    return natsorted(out)


def _find_ckpts(exp_dir: str) -> List[str]:
    exts = (".pth.tar", ".pth", ".tar")
    out = []
    for name in os.listdir(exp_dir):
        p = os.path.join(exp_dir, name)
        if os.path.isfile(p) and name.endswith(exts):
            out.append(p)
    return natsorted(out)


def _csv_has_hd95(path: str) -> bool:
    """Return True iff per_case.csv header contains all required new columns."""
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            hdr = next(csv.reader(f), [])
        return all(c in hdr for c in _REQUIRED_COLS)
    except Exception:
        return False


@dataclass
class EvalResult:
    exp_name: str
    ckpt: str
    n_cases: int
    dice_group_mean_mean: float
    dice_group_mean_std: float
    non_jec_mean: float
    non_jec_std: float
    sdlogj_mean: float
    sdlogj_std: float
    hd95_mean: float
    hd95_std: float
    assd_mean: float
    assd_std: float


def eval_one_checkpoint(
    ckpt_path: str,
    out_dir: str,
    ixi_root: str,
    test_subdir: str = "Test",
    max_cases: Optional[int] = None,
    device: Optional[str] = None,
    resume: bool = True,
) -> EvalResult:
    _setup_transmorph_path()
    import utils as tm_utils  # noqa: WPS433
    from data import datasets, trans
    from metrics_full import jacobian_stats as jacobian_stats_flow
    from metrics_full import per_label_dice

    exp_name = os.path.basename(os.path.dirname(ckpt_path))
    os.makedirs(out_dir, exist_ok=True)
    per_case_csv = os.path.join(out_dir, "per_case.csv")
    summary_json = os.path.join(out_dir, "summary.json")

    label_names = _label_names_0_45()
    group_map = _build_group_indices(label_names)

    atlas = _resolve_atlas(ixi_root)
    test_dir = os.path.join(ixi_root, test_subdir)
    pkl_list = natsorted(
        [os.path.join(test_dir, p) for p in os.listdir(test_dir) if p.endswith(".pkl")]
    )
    if not pkl_list:
        raise FileNotFoundError(f"No test .pkl under {test_dir}")
    if max_cases is not None:
        pkl_list = pkl_list[: int(max_cases)]

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = dev == "cuda"
    torch_device = torch.device(dev)

    from models.TransMorph import CONFIGS as CONFIGS_TM  # noqa: WPS433
    import models.TransMorph as TransMorph  # noqa: WPS433

    config = CONFIGS_TM["TransMorph"]
    model = TransMorph.TransMorph(config)
    print(f"[Ablation-Eval] loading ckpt: {ckpt_path}", flush=True)
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = st["state_dict"] if isinstance(st, dict) and "state_dict" in st else st
    model.load_state_dict(sd)
    model.to(torch_device)
    model.eval()
    reg = tm_utils.register_model(config.img_size, "bilinear")
    reg = reg.to(torch_device)

    test_tf = transforms.Compose([trans.Seg_norm(), trans.NumpyType((np.float32, np.int16))])
    ds = datasets.IXIBrainInferDataset(pkl_list, atlas, transforms=test_tf)

    # Resume: only reuse existing per_case.csv if it has the new HD95/ASSD columns.
    done = 0
    if resume and os.path.isfile(per_case_csv):
        if _csv_has_hd95(per_case_csv):
            with open(per_case_csv, "r", encoding="utf-8", newline="") as f:
                done = max(0, len(list(csv.reader(f))) - 1)
            print(f"[Ablation-Eval] resuming {exp_name} from case {done}", flush=True)
        else:
            print(
                f"[Ablation-Eval] per_case.csv missing HD95/ASSD columns – restarting {exp_name}",
                flush=True,
            )
            os.remove(per_case_csv)
            if os.path.isfile(summary_json):
                os.remove(summary_json)
            done = 0

    loader = DataLoader(
        torch.utils.data.Subset(ds, list(range(done, len(ds)))) if done > 0 else ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
        drop_last=True,
    )

    header = [
        "case_idx",
        "pkl",
        "dice_group_mean",
        "non_jec",
        "SDlogJ",
        "HD95_mean",
        "ASSD_mean",
    ] + [f"dice_{g}" for g in OUTSTRUCT]

    if done == 0:
        with open(per_case_csv, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(header)

    dice_group_means: List[float] = []
    non_jecs: List[float] = []
    sdlogjs: List[float] = []
    hd95s: List[float] = []
    assds: List[float] = []

    t0 = time.time()
    start_idx = done
    with torch.no_grad():
        for local_i, data in enumerate(loader):
            case_idx = start_idx + local_i
            x, y, x_seg, y_seg = [t.to(torch_device) for t in data]
            x_in = torch.cat((x, y), dim=1)
            x_def, flow = model(x_in)

            # warp seg via one-hot (identical to eval_any)
            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=46)
            x_seg_oh = torch.squeeze(x_seg_oh, 1).permute(0, 4, 1, 2, 3).contiguous()
            x_segs = []
            for i in range(46):
                x_segs.append(reg([x_seg_oh[:, i : i + 1].float(), flow.float()]))
            x_segs = torch.cat(x_segs, dim=1)
            def_out = torch.argmax(x_segs, dim=1, keepdim=True)

            pred = def_out.long().cpu().numpy()[0, 0]
            true = y_seg.long().cpu().numpy()[0, 0]

            dices, _ = per_label_dice(pred, true, num_classes=46)
            gvals, gmean = grouped_dice_from_per_label(dices.astype(np.float64), group_map)

            flow_np = flow.detach().cpu().numpy()[0, ...]
            jst = jacobian_stats_flow(flow_np)
            non_jec = float(jst["non_jec"])
            sdlogj = float(jst["SDlogJ"])

            hd95, assd = grouped_hd95_assd(pred, true, group_map, spacing=1.0)

            pkl_basename = os.path.splitext(os.path.basename(pkl_list[case_idx]))[0]
            with open(per_case_csv, "a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(
                    [
                        case_idx,
                        pkl_basename,
                        f"{gmean:.8f}",
                        f"{non_jec:.8f}",
                        f"{sdlogj:.8f}",
                        f"{hd95:.6f}" if np.isfinite(hd95) else "nan",
                        f"{assd:.6f}" if np.isfinite(assd) else "nan",
                    ]
                    + [f"{v:.8f}" if np.isfinite(v) else "nan" for v in gvals.tolist()]
                )

            dice_group_means.append(gmean)
            non_jecs.append(non_jec)
            sdlogjs.append(sdlogj)
            hd95s.append(hd95)
            assds.append(assd)

            if (case_idx + 1) % 5 == 0 or case_idx == 0:
                print(
                    f"[Ablation-Eval] {exp_name}  {case_idx + 1}/{len(pkl_list)}  "
                    f"diceG={gmean:.4f} non_jec={non_jec:.6f} SDlogJ={sdlogj:.4f} "
                    f"HD95={hd95:.3f} ASSD={assd:.4f}",
                    flush=True,
                )

    # Aggregate over all rows (including any resumed portion)
    all_dice, all_nonjec, all_sdlogj, all_hd95, all_assd = [], [], [], [], []
    with open(per_case_csv, "r", encoding="utf-8", newline="") as f:
        for row in list(csv.reader(f))[1:]:
            all_dice.append(float(row[2]))
            all_nonjec.append(float(row[3]))
            all_sdlogj.append(float(row[4]))
            v_hd = row[5]
            v_as = row[6]
            all_hd95.append(float(v_hd) if v_hd != "nan" else float("nan"))
            all_assd.append(float(v_as) if v_as != "nan" else float("nan"))

    def _stat(vals: List[float]) -> Tuple[float, float]:
        a = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
        if a.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(a)), float(np.std(a, ddof=0))

    d_m, d_s = _stat(all_dice)
    j_m, j_s = _stat(all_nonjec)
    sl_m, sl_s = _stat(all_sdlogj)
    hd_m, hd_s = _stat(all_hd95)
    as_m, as_s = _stat(all_assd)

    summ = {
        "exp_name": exp_name,
        "ckpt": ckpt_path,
        "n_cases": len(all_dice),
        "dice_group_mean": {"mean": d_m, "std": d_s},
        "non_jec": {"mean": j_m, "std": j_s},
        "SDlogJ": {"mean": sl_m, "std": sl_s},
        "HD95_mean": {"mean": hd_m, "std": hd_s},
        "ASSD_mean": {"mean": as_m, "std": as_s},
        "elapsed_s": float(time.time() - t0),
        "device": str(torch_device),
    }
    with open(summary_json, "w", encoding="utf-8") as jf:
        json.dump(summ, jf, indent=2)
    print(
        f"[Ablation-Eval] done {exp_name}  elapsed_s={summ['elapsed_s']:.1f}  "
        f"Dice={d_m:.4f} HD95={hd_m:.3f} ASSD={as_m:.4f}",
        flush=True,
    )

    return EvalResult(
        exp_name=exp_name,
        ckpt=ckpt_path,
        n_cases=len(all_dice),
        dice_group_mean_mean=d_m,
        dice_group_mean_std=d_s,
        non_jec_mean=j_m,
        non_jec_std=j_s,
        sdlogj_mean=sl_m,
        sdlogj_std=sl_s,
        hd95_mean=hd_m,
        hd95_std=hd_s,
        assd_mean=as_m,
        assd_std=as_s,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", default=_abs("experiments"), help="repo experiments root")
    ap.add_argument("--prefix", default="TransMorph_IXI_HER_", help="experiment dir name prefix")
    ap.add_argument("--ckpt_strategy", default="best_dsc", choices=["best_dsc", "latest"])
    ap.add_argument("--results_root", default=_abs("IXI/Eval_Results/ablation_batch"))
    ap.add_argument("--ixi_root", default=_abs("IXI_data"))
    ap.add_argument("--test_subdir", default="Test")
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument("--device", default=None, choices=[None, "cuda", "cpu"], nargs="?")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no_resume", action="store_true", default=False)
    args = ap.parse_args()

    resume = bool(args.resume) and (not bool(args.no_resume))
    exp_root = _abs(args.exp_root)
    out_root = _abs(args.results_root)
    os.makedirs(out_root, exist_ok=True)

    exp_dirs = _find_experiments(exp_root, args.prefix)
    if not exp_dirs:
        print(f"No experiments found under {exp_root} with prefix {args.prefix}")
        return 2

    # Global summary CSV – always rewrite header for clean run
    summary_csv = os.path.join(out_root, "summary.csv")
    _GLOB_HEADER = [
        "exp_name", "ckpt", "n_cases",
        "dice_group_mean_mean", "dice_group_mean_std",
        "non_jec_mean", "non_jec_std",
        "SDlogJ_mean", "SDlogJ_std",
        "HD95_mean", "HD95_std",
        "ASSD_mean", "ASSD_std",
    ]
    need_header = not os.path.isfile(summary_csv)
    if not need_header:
        # Check if existing summary.csv has HD95/ASSD columns
        try:
            with open(summary_csv, "r", encoding="utf-8", newline="") as f:
                existing_hdr = next(csv.reader(f), [])
            if not all(c in existing_hdr for c in ("HD95_mean", "ASSD_mean")):
                need_header = True
        except Exception:
            need_header = True
    if need_header:
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(_GLOB_HEADER)

    for exp_dir in exp_dirs:
        ckpts = _find_ckpts(exp_dir)
        if not ckpts:
            print(f"[Ablation-Eval] skip empty: {exp_dir}", flush=True)
            continue
        ckpt = _select_ckpt(ckpts, args.ckpt_strategy)
        exp_name = os.path.basename(exp_dir)
        exp_out = os.path.join(out_root, exp_name)

        # Skip only when summary.json has HD95 column AND n_cases == full test size
        summ_p = os.path.join(exp_out, "summary.json")
        per_case_p = os.path.join(exp_out, "per_case.csv")
        if resume and os.path.isfile(summ_p) and args.max_cases is None:
            try:
                with open(summ_p, "r", encoding="utf-8") as f:
                    s = json.load(f)
                done_cases = int(s.get("n_cases", -1))
                has_hd = "HD95_mean" in s
            except Exception:
                done_cases, has_hd = -1, False
            test_d = os.path.join(_abs(args.ixi_root), args.test_subdir)
            full = len([p for p in os.listdir(test_d) if p.endswith(".pkl")]) if os.path.isdir(test_d) else -1
            if full > 0 and done_cases == full and has_hd and _csv_has_hd95(per_case_p):
                print(f"[Ablation-Eval] already done (with HD95), skip: {exp_name}", flush=True)
                continue

        res = eval_one_checkpoint(
            ckpt_path=ckpt,
            out_dir=exp_out,
            ixi_root=_abs(args.ixi_root),
            test_subdir=args.test_subdir,
            max_cases=args.max_cases,
            device=args.device,
            resume=resume,
        )

        with open(summary_csv, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([
                res.exp_name,
                res.ckpt,
                res.n_cases,
                f"{res.dice_group_mean_mean:.8f}",
                f"{res.dice_group_mean_std:.8f}",
                f"{res.non_jec_mean:.8f}",
                f"{res.non_jec_std:.8f}",
                f"{res.sdlogj_mean:.8f}",
                f"{res.sdlogj_std:.8f}",
                f"{res.hd95_mean:.6f}" if np.isfinite(res.hd95_mean) else "nan",
                f"{res.hd95_std:.6f}" if np.isfinite(res.hd95_std) else "nan",
                f"{res.assd_mean:.6f}" if np.isfinite(res.assd_mean) else "nan",
                f"{res.assd_std:.6f}" if np.isfinite(res.assd_std) else "nan",
            ])

    print(f"[Ablation-Eval] wrote global summary: {summary_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
