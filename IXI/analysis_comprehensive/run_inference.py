"""
Single-model inference + metric CSVs and summary.json.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
import time
import inspect
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from natsort import natsorted
from torch.utils.data import DataLoader
from torchvision import transforms

from . import metrics as M
from .config import (
    ModelSpec,
    ATLAS_PKL,
    DATA_ROOT_DEFAULT,
    TRANSMORPH_DIR,
    get_ckpt_dir,
)


def _setup_transmorph_path() -> None:
    if TRANSMORPH_DIR not in sys.path:
        sys.path.insert(0, TRANSMORPH_DIR)


def _resolve_checkpoint(ckpt_dir: str, spec: ModelSpec) -> str:
    if spec.ckpt_file:
        p = os.path.join(ckpt_dir, spec.ckpt_file)
        if os.path.isfile(p):
            return p
    exts = (".pth.tar", ".pth", ".tar")
    files: List[str] = []
    for f in os.listdir(ckpt_dir):
        if f.startswith(".") or os.path.isdir(os.path.join(ckpt_dir, f)):
            continue
        if f.endswith(exts):
            files.append(f)
    if not files:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
    return os.path.join(ckpt_dir, natsorted(files)[-1])


def _load_state(ckpt_path: str, map_loc: torch.device) -> Any:
    kw = {"map_location": map_loc}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kw["weights_only"] = False
    return torch.load(ckpt_path, **kw)


def _lncc_and_ssim_torch(
    warped: torch.Tensor,
    fixed: torch.Tensor,
) -> Tuple[float, float]:
    """warped, fixed: (1,1,D,H,W) on CUDA."""
    _setup_transmorph_path()
    from losses import NCC_vxm, ssim3D

    ncc = NCC_vxm(win=[9, 9, 9])
    with torch.no_grad():
        ncc_l = ncc(fixed, warped)  # negative mean cc
        lncc = float((-ncc_l).item())
        s = float(
            ssim3D(
                warped, fixed, window_size=11, size_average=True
            ).item()
        )
    return lncc, s


def run_one_model(
    spec: ModelSpec,
    gpu_id: int,
    out_root: str,
    test_dir: Optional[str] = None,
    atlas_path: Optional[str] = None,
    limit: Optional[int] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    Run one model, write to out_root / spec.name /.
    Returns result dict (empty on failure after logging).
    """
    _setup_transmorph_path()
    import utils
    from data import datasets, trans
    from voi_definitions import VOI_LBLS

    test_p = test_dir or os.path.join(DATA_ROOT_DEFAULT, "Test/")
    atlas = atlas_path or ATLAS_PKL
    mdir = get_ckpt_dir(spec)
    out_dir = os.path.join(out_root, spec.name)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    err_path = os.path.join(out_dir, "_ERROR.log")
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)

    try:
        ckpt = _resolve_checkpoint(mdir, spec)
    except Exception as e:
        with open(err_path, "w") as ef:
            ef.write(traceback.format_exc())
        print(f"[{spec.name}] FAIL resolve ckpt: {e}")
        return {"error": str(e), "model": spec.name}

    try:
        if spec.backbone == "TransMorphBayes":
            from models.TransMorph_Bayes import CONFIGS as Cfg
            import models.TransMorph_Bayes as TM

            config = Cfg["TransMorphBayes"]
            model = TM.TransMorphBayes(config)
        else:
            from models.TransMorph import CONFIGS as Cfg
            import models.TransMorph as TM

            config = Cfg["TransMorph"]
            model = TM.TransMorph(config)
        st = _load_state(ckpt, device)
        model.load_state_dict(st["state_dict"] if "state_dict" in st else st)
        model.to(device)
        model.eval()
        reg = utils.register_model(config.img_size, "bilinear").to(device)
    except Exception:
        with open(err_path, "w") as ef:
            ef.write(traceback.format_exc())
        print(f"[{spec.name}] FAIL load model")
        return {"error": "load", "model": spec.name}

    test_tf = transforms.Compose(
        [trans.Seg_norm(), trans.NumpyType((np.float32, np.int16))]
    )
    test_p = (test_p or os.path.join(DATA_ROOT_DEFAULT, "Test")).rstrip("/") + "/"
    paths = natsorted(glob.glob(os.path.join(test_p, "*.pkl")))
    if limit is not None:
        paths = paths[: int(limit)]
    n_total = len(paths)
    if n_total == 0:
        with open(err_path, "w") as ef:
            ef.write("no test pkl\n")
        return {"error": "no data", "model": spec.name}

    voi_hdr = [str(x) for x in VOI_LBLS]
    w_header = ["subject"] + voi_hdr

    def _open_csv(name: str) -> str:
        p = os.path.join(out_dir, name)
        return p

    n_done = 0
    if resume and os.path.isfile(_open_csv("dice.csv")):
        with open(_open_csv("dice.csv"), "r", encoding="utf-8", newline="") as f:
            n_done = max(0, len(f.readlines()) - 1)
    paths = paths[n_done:]

    if n_done == 0:
        for fname in [
            "dice.csv",
            "jaccard.csv",
            "vs.csv",
            "hd95.csv",
            "assd.csv",
            "nsd1mm.csv",
            "intensity.csv",
            "jacobian.csv",
            "runtime.csv",
        ]:
            pth = _open_csv(fname)
            if os.path.isfile(pth):
                os.remove(pth)

    int_hdr = [
        "subject",
        "ncc",
        "lncc",
        "mi",
        "nmi",
        "ssim3d",
        "mse",
        "psnr",
    ]
    jac_hdr = [
        "subject",
        "non_jac_frac",
        "SDlogJ",
        "J_min",
        "J_p01",
        "J_p05",
        "J_p50",
        "J_p95",
        "J_p99",
        "J_max",
        "bending_energy",
        "mean_abs_div",
    ]
    run_hdr = ["subject", "runtime_s", "peak_mem_GB"]

    if n_done == 0:
        with open(_open_csv("dice.csv"), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(w_header)
        for fn, h in [
            ("jaccard.csv", w_header),
            ("vs.csv", w_header),
            ("hd95.csv", w_header),
            ("assd.csv", w_header),
            ("nsd1mm.csv", w_header),
        ]:
            with open(_open_csv(fn), "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(h)
        with open(_open_csv("intensity.csv"), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(int_hdr)
        with open(_open_csv("jacobian.csv"), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(jac_hdr)
        with open(_open_csv("runtime.csv"), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(run_hdr)

    n_sub = n_total
    loader: Optional[DataLoader] = None
    if paths:
        ds = datasets.IXIBrainInferDataset(paths, atlas, transforms=test_tf)
        loader = DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True
        )

    logJ_chunks: List[np.ndarray] = []
    subj = n_done

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with torch.no_grad():
        for data in loader or []:
            x, y, x_seg, y_seg = [t.to(device) for t in data]
            t0 = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            x_in = torch.cat((x, y), dim=1)

            if spec.backbone == "TransMorphBayes":
                _, flow_list, errs = utils.get_mc_preds_w_errors(
                    model, x_in, y, mc_iter=spec.mc_iter
                )
                flow = flow_list[int(np.argmin(errs))]
                x_def_t = reg([x, flow])
            else:
                x_def_t, flow = model(x_in)

            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=46)
            x_seg_oh = x_seg_oh.squeeze(1).permute(0, 4, 1, 2, 3).contiguous()
            x_segs = []
            for i in range(46):
                x_segs.append(reg([x_seg_oh[:, i : i + 1].float(), flow]))
            x_segs = torch.cat(x_segs, 1)
            def_out = torch.argmax(x_segs, 1, keepdim=True)

            runtime = time.perf_counter() - t0
            if device.type == "cuda":
                pm = torch.cuda.max_memory_allocated(device) / 1e9
            else:
                pm = 0.0

            pred_np = def_out.detach().cpu().numpy()[0, 0, ...]
            y_np = y_seg.detach().cpu().numpy()[0, 0, ...]
            warped_np = x_def_t.detach().cpu().numpy()[0, 0, ...]
            fixed_np = y.detach().cpu().numpy()[0, 0, ...]
            flow_np = flow.detach().cpu().numpy()[0, ...]

            dice_d, jacc_d, vs_d = M.overlap_per_voi(pred_np, y_np, VOI_LBLS)
            hd_d, as_d, ns_d = M.surface_per_voi(
                pred_np, y_np, VOI_LBLS, tau=1.0, spacing=M.DEFAULT_SPACING
            )

            ncc = M.ncc_global_numpy(warped_np, fixed_np)
            mi, nmi = M.mutual_info_and_nmi(warped_np, fixed_np, bins=64)
            mse, psnr = M.mse_psnr(warped_np, fixed_np)
            lncc, ssim3 = _lncc_and_ssim_torch(x_def_t, y)

            jac = utils.jacobian_determinant_vxm(flow_np)
            j_st = M.jacobian_field_stats(jac)
            be = M.bending_energy_full(flow_np, M.DEFAULT_SPACING)
            madv = M.mean_abs_divergence_np(flow_np)
            logJ_samples = M.subsample_log_j_pos(jac, max_voxels=200_000, seed=subj)
            if logJ_samples.size:
                logJ_chunks.append(logJ_samples)

            subj_id = f"p_{subj}"

            def _row_d(d: Dict[int, float]) -> List[str]:
                return [f"{d.get(l, float('nan')):.8f}" for l in VOI_LBLS]

            with open(_open_csv("dice.csv"), "a", newline="") as f:
                csv.writer(f).writerow([subj_id] + _row_d(dice_d))
            with open(_open_csv("jaccard.csv"), "a", newline="") as f:
                csv.writer(f).writerow([subj_id] + _row_d(jacc_d))
            with open(_open_csv("vs.csv"), "a", newline="") as f:
                csv.writer(f).writerow([subj_id] + _row_d(vs_d))
            with open(_open_csv("hd95.csv"), "a", newline="") as f:
                csv.writer(f).writerow([subj_id] + _row_d(hd_d))
            with open(_open_csv("assd.csv"), "a", newline="") as f:
                csv.writer(f).writerow([subj_id] + _row_d(as_d))
            with open(_open_csv("nsd1mm.csv"), "a", newline="") as f:
                csv.writer(f).writerow([subj_id] + _row_d(ns_d))

            with open(_open_csv("intensity.csv"), "a", newline="") as f:
                csv.writer(f).writerow(
                    [subj_id, f"{ncc:.8f}", f"{lncc:.8f}", f"{mi:.8f}", f"{nmi:.8f}", f"{ssim3:.8f}", f"{mse:.8e}", f"{psnr:.4f}"]
                )
            with open(_open_csv("jacobian.csv"), "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        subj_id,
                        f"{j_st['non_jac_frac']:.8f}",
                        f"{j_st['SDlogJ']:.8f}",
                        f"{j_st['J_min']:.8f}",
                        f"{j_st['J_p01']:.8f}",
                        f"{j_st['J_p05']:.8f}",
                        f"{j_st['J_p50']:.8f}",
                        f"{j_st['J_p95']:.8f}",
                        f"{j_st['J_p99']:.8f}",
                        f"{j_st['J_max']:.8f}",
                        f"{be:.8f}",
                        f"{madv:.8f}",
                    ]
                )
            with open(_open_csv("runtime.csv"), "a", newline="") as f:
                csv.writer(f).writerow(
                    [subj_id, f"{runtime:.6f}", f"{pm:.4f}"]
                )

            if subj == 0:
                J_neg = (jac <= 0).astype(np.float32)
                D, H, W = fixed_np.shape
                sl = [D // 2, H // 2, W // 2]
                titles = [("axial", 0, sl[0]), ("coronal", 1, sl[1]), ("sagittal", 2, sl[2])]
                for tag, ad, idx in titles:
                    if ad == 0:
                        sli = J_neg[idx, :, :]
                        fsl = np.rot90(fixed_np[idx, :, :], k=0)
                    elif ad == 1:
                        sli = J_neg[:, idx, :]
                        fsl = np.rot90(fixed_np[:, idx, :], k=0)
                    else:
                        sli = J_neg[:, :, idx]
                        fsl = np.rot90(fixed_np[:, :, idx], k=0)
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                    ax.imshow(fsl, cmap="gray")
                    o = np.zeros((*sli.shape, 4), dtype=np.float32)
                    o[..., 0] = 1.0
                    o[..., 3] = sli * 0.7
                    ax.imshow(o)
                    ax.set_title(f"folding J<=0 (red) {tag}")
                    ax.axis("off")
                    fig.tight_layout()
                    fig.savefig(os.path.join(fig_dir, f"folding_map_{tag}.png"), dpi=120)
                    plt.close(fig)

            subj += 1

    if logJ_chunks:
        all_log = np.concatenate(logJ_chunks) if len(logJ_chunks) > 1 else logJ_chunks[0]
        if all_log.size > 500_000:
            all_log = all_log[:: (all_log.size // 500_000 + 1)]
        np.save(os.path.join(out_dir, "jacobian_log_samples.npy"), all_log)

    summ: Dict[str, Any] = {
        "model": spec.name,
        "ckpt": ckpt,
        "n_subjects": n_sub,
        "per_subject": {},
    }

    for csv_name, key_cols in [
        ("dice.csv", "dice_voi"),
        ("jaccard.csv", "jaccard_voi"),
        ("vs.csv", "vs_voi"),
        ("hd95.csv", "hd95_voi"),
        ("assd.csv", "assd_voi"),
        ("nsd1mm.csv", "nsd1mm_voi"),
    ]:
        p = _open_csv(csv_name)
        with open(p) as f:
            rows = list(csv.reader(f))
        hdr, body = rows[0], rows[1:]
        vals: List[List[float]] = []
        for row in body:
            vals.append([float(x) if x and x.lower() != "nan" else float("nan") for x in row[1:]])
        arr = np.asarray(vals, dtype=np.float64) if body else np.zeros((0, len(VOI_LBLS)))
        if arr.size and arr.shape[0]:
            rmean = np.nanmean(arr, axis=1)
            summ["per_subject"].setdefault(key_cols, {})
            summ["per_subject"][key_cols] = {
                "across_voi_per_subject": {
                    "mean": float(np.nanmean(rmean)),
                    "std": float(np.nanstd(rmean, ddof=0)),
                }
            }
        else:
            summ["per_subject"][key_cols] = {
                "across_voi_per_subject": {"mean": float("nan"), "std": float("nan")}
            }

    p_int = _open_csv("intensity.csv")
    with open(p_int) as f:
        ir = list(csv.reader(f))
    for ci, cname in enumerate(
        ["ncc", "lncc", "mi", "nmi", "ssim3d", "mse", "psnr"]
    ):
        col = [float(x) for x in [r[ci + 1] for r in ir[1:]]]
        summ["per_subject"][cname] = {
            "mean": float(np.mean(col)) if col else float("nan"),
            "std": float(np.std(col, ddof=0)) if col else float("nan"),
        }
    p_j = _open_csv("jacobian.csv")
    with open(p_j) as f:
        jr = list(csv.reader(f))
    jnames = [
        "non_jac_frac",
        "SDlogJ",
        "J_min",
        "J_p01",
        "J_p05",
        "J_p50",
        "J_p95",
        "J_p99",
        "J_max",
        "bending_energy",
        "mean_abs_div",
    ]
    for ji, jn in enumerate(jnames):
        col = [float(x) for x in [r[ji + 1] for r in jr[1:]]]
        summ["per_subject"][jn] = {
            "mean": float(np.mean(col)) if col else float("nan"),
            "std": float(np.std(col, ddof=0)) if col else float("nan"),
        }
    p_rt = _open_csv("runtime.csv")
    with open(p_rt) as f:
        rt = list(csv.reader(f))
    rt_t = [float(x) for x in [r[1] for r in rt[1:]]]
    rt_m = [float(x) for x in [r[2] for r in rt[1:]]]
    summ["inference"] = {
        "runtime_s_per_pair": {
            "mean": float(np.mean(rt_t)) if rt_t else 0.0,
            "std": float(np.std(rt_t, ddof=0)) if rt_t else 0.0,
        },
        "peak_mem_GB": {
            "max": float(np.max(rt_m)) if rt_m else 0.0,
            "mean": float(np.mean(rt_m)) if rt_m else 0.0,
        },
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(f"[{spec.name}] done, wrote {n_sub} subjects to {out_dir}")
    return summ
