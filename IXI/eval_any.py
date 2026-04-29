# -*- coding: utf-8 -*-
"""
Per-pair inference + full metrics; writes per_case.csv, flow_stats.json, meta.json.
"""
from __future__ import annotations

import csv
import importlib
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from natsort import natsorted
from torch.utils.data import DataLoader
from torchvision import transforms

IXI_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(IXI_DIR)


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(REPO_ROOT, p))


def _setup_transmorph_path():
    tm = os.path.join(IXI_DIR, "TransMorph")
    if tm not in sys.path:
        sys.path.insert(0, tm)


def load_cfg(path: Optional[str] = None) -> dict:
    p = path or os.path.join(IXI_DIR, "eval_configs.yaml")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_atlas(ixi_root: str, atlas: Optional[str]) -> str:
    if atlas and os.path.isfile(_abs(atlas)):
        return _abs(atlas)
    for name in ("atlas.pkl", "altas.pkl"):
        ap = os.path.join(ixi_root, name)
        if os.path.isfile(ap):
            return ap
    raise FileNotFoundError(f"No atlas in {ixi_root}")


def run_one_model(
    model_entry: dict,
    global_cfg: dict,
    pkl_glob: Optional[str] = None,
    max_cases: Optional[int] = None,
) -> str:
    """
    Run evaluation for a single model. Returns path to out_dir.
    """
    _setup_transmorph_path()
    import utils as tm_utils
    from data import datasets, trans
    from losses import NCC_vxm

    m_id = model_entry["id"]
    out_root = _abs(global_cfg.get("results_dir", "IXI/Eval_Results"))
    out_dir = os.path.join(out_root, m_id)
    os.makedirs(out_dir, exist_ok=True)

    ixi = _abs(global_cfg.get("ixi_root", "IXI_data"))
    test_sub = global_cfg.get("test_subdir", "Test")
    atlas = resolve_atlas(ixi, global_cfg.get("atlas"))
    test_glob = pkl_glob or os.path.join(ixi, test_sub, "*.pkl")
    pkl_list = natsorted(glob.glob(test_glob))
    if not pkl_list:
        raise FileNotFoundError(f"No test .pkl under {test_glob}")
    if max_cases is not None:
        pkl_list = pkl_list[: int(max_cases)]

    n_total = len(pkl_list)
    print(
        f"[IXI-Eval] {m_id}  test_cases={n_total}  out={out_dir}",
        flush=True,
    )

    spacing = float(np.asarray(global_cfg.get("spacing", [1, 1, 1]))[0])
    ssim_win = int(global_cfg.get("ssim_win", 7))
    nmi_bins = int(global_cfg.get("nmi_bins", 64))
    n_bins = int(global_cfg.get("jacobian_hist_bins", 100))
    num_lbl = int(global_cfg.get("num_labels", 46))
    compute_ice = bool(global_cfg.get("compute_ice", False))

    # Adapter
    if IXI_DIR not in sys.path:
        sys.path.insert(0, IXI_DIR)
    ad_name = model_entry.get("adapter", m_id)
    ad = importlib.import_module(f"adapters.{ad_name}")
    pref = getattr(ad, "PREFERRED_DEVICE", None)
    if pref is not None:
        device = str(pref).lower()
        if device not in ("cuda", "cpu"):
            raise ValueError(f"Unsupported PREFERRED_DEVICE from adapter {ad_name}: {pref}")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = device == "cuda"
    model, config = ad.build_model(device=device)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    model.eval()
    reg = tm_utils.register_model(config.img_size, "bilinear")
    if use_cuda:
        reg.cuda()
    ncc = NCC_vxm()
    if use_cuda:
        ncc.cuda()

    from metrics_full import (  # noqa: WPS433
        hd95_assd_mean_over_labels,
        jacobian_stats,
        jaccard_mean_over_labels,
        nmi_3d,
        per_label_dice,
        ssim3d,
        volume_similarity,
    )

    test_composed = transforms.Compose(
        [trans.Seg_norm(), trans.NumpyType((np.float32, np.int16))]
    )
    test_set = datasets.IXIBrainInferDataset(
        pkl_list, atlas, transforms=test_composed
    )
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=use_cuda, drop_last=True
    )

    _tm = os.path.join(IXI_DIR, "TransMorph")
    _cwd = os.getcwd()
    os.chdir(_tm)
    try:
        label_dict = tm_utils.process_label()
    finally:
        os.chdir(_cwd)
    per_case_path = os.path.join(out_dir, "per_case.csv")
    with open(per_case_path, "w", newline="", encoding="utf-8") as f:
        w = None
        flow_rows: List[dict] = []
        stdy = 0
        t0 = time.time()
        for data in test_loader:
            t_case0 = time.time()
            data_t = [t.to(device) for t in data]
            x, y, x_seg, y_seg = data_t
            t_inf0 = time.time()
            with torch.no_grad():
                x_def, flow = ad.forward(model, x, y)
            flow_b = None
            if compute_ice:
                with torch.no_grad():
                    _y_def, flow_b = ad.forward(model, y, x)
            t_inf1 = time.time() - t_inf0

            t_warp0 = time.time()
            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=num_lbl)
            x_seg_oh = torch.squeeze(x_seg_oh, 1)
            x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            x_segs = []
            for i in range(num_lbl):
                def_seg = reg(
                    [x_seg_oh[:, i : i + 1, ...].float(), flow.float()]
                )
                x_segs.append(def_seg)
            x_segs = torch.cat(x_segs, dim=1)
            def_out = torch.argmax(x_segs, dim=1, keepdim=True)
            del x_segs, x_seg_oh
            t_warp1 = time.time() - t_warp0

            t_metric0 = time.time()
            y_np = y.detach().cpu().numpy()[0, 0]
            xw = x_def.detach().cpu().numpy()[0, 0]
            pred = def_out.long().cpu().numpy()[0, 0, ...]
            true = y_seg.long().cpu().numpy()[0, 0, ...]
            dices, _ = per_label_dice(pred, true, num_classes=num_lbl)
            vs = volume_similarity(pred, true, num_classes=num_lbl)
            jacc = jaccard_mean_over_labels(pred, true, num_classes=num_lbl)
            d_mean = float(np.mean(dices))
            h_mean, a_mean = hd95_assd_mean_over_labels(
                pred, true, num_classes=num_lbl, spacing=spacing
            )
            wdim = int(min(
                ssim_win,
                xw.shape[0],
                xw.shape[1],
                xw.shape[2],
            ))
            if wdim % 2 == 0:
                wdim = max(3, wdim - 1)
            ssim_v = ssim3d(
                xw, y_np, data_range=1.0, win_size=wdim
            )
            nmi = nmi_3d(y_np, xw, bins=nmi_bins)
            if use_cuda:
                with torch.no_grad():
                    # NCC_vxm forward returns -mean(CC). Report +mean(CC) as LNCC.
                    # (B,1,D,H,W) — do not add extra batch dim (would break NCC 3D)
                    lncc = float(-ncc(y, x_def).cpu().item())
            else:
                # CPU classical adapters (e.g., SyN/Affine): avoid CUDA-hardcoded NCC_vxm.
                lncc = float(M.ncc_global_numpy(y_np, xw))

            f_np = flow.detach().cpu().numpy()[0, ...]
            jstat = jacobian_stats(f_np, n_bins=n_bins)
            non_jec = jstat["non_jec"]
            peak_mb = 0.0
            if use_cuda:
                peak_mb = (
                    torch.cuda.max_memory_allocated() / 1024.0**3
                )
            if compute_ice and flow_b is not None:
                from metrics_full import inverse_consistency_error  # noqa: WPS433

                fb = flow_b.detach().cpu().numpy()[0, ...]
                ice = inverse_consistency_error(
                    f_np, fb, spacing=global_cfg.get("spacing")
                )
            else:
                ice = float("nan")
            t_metric1 = time.time() - t_metric0

            pkl_basename = os.path.splitext(
                os.path.basename(pkl_list[stdy])
            )[0]
            if w is None:
                hcols = [f"dice_{i}" for i in range(num_lbl)]
                hcols += [f"vs_{i}" for i in range(num_lbl)]
                hcols += [
                    "jaccard_mean",
                    "dice_mean",
                    "HD95_mean",
                    "ASSD_mean",
                    "SSIM",
                    "NMI",
                    "LNCC",
                    "non_jec",
                    "SDlogJ",
                    "J_p01",
                    "J_p50",
                    "J_p99",
                    "J_min",
                    "J_max",
                    "inference_s",
                    "peak_mem_gb",
                    "ICE",
                    "pkl",
                    "stdy_idx",
                ]
                w = csv.DictWriter(f, fieldnames=hcols, extrasaction="ignore")
                w.writeheader()

            row = {f"dice_{i}": dices[i] for i in range(num_lbl)}
            row.update({f"vs_{i}": vs[i] for i in range(num_lbl)})
            row.update(
                {
                    "jaccard_mean": jacc,
                    "dice_mean": d_mean,
                    "HD95_mean": h_mean,
                    "ASSD_mean": a_mean,
                    "SSIM": ssim_v,
                    "NMI": nmi,
                    "LNCC": lncc,
                    "non_jec": non_jec,
                    "SDlogJ": jstat["SDlogJ"],
                    "J_p01": jstat["J_p01"],
                    "J_p50": jstat["J_p50"],
                    "J_p99": jstat["J_p99"],
                    "J_min": jstat["J_min"],
                    "J_max": jstat["J_max"],
                    "inference_s": t_inf1,
                    "peak_mem_gb": peak_mb,
                    "ICE": ice,
                    "pkl": pkl_basename,
                    "stdy_idx": stdy,
                }
            )
            w.writerow(row)
            f.flush()
            print(
                f"[IXI-Eval] {m_id}  {stdy + 1}/{n_total}  {pkl_basename}  "
                f"inf_s={t_inf1:.1f}  warp_s={t_warp1:.1f}  cpuMetric_s={t_metric1:.1f}  "
                f"case_s={time.time() - t_case0:.1f}  dice_mean={d_mean:.4f}  non_jec={non_jec:.6f}",
                flush=True,
            )
            flow_rows.append(
                {
                    "pkl": pkl_basename,
                    "stdy_idx": stdy,
                    "hist_counts": jstat["hist_counts"],
                    "hist_edges": jstat["hist_edges"],
                    "non_jec": non_jec,
                }
            )
            stdy += 1

    total_t = time.time() - t0
    with open(
        os.path.join(out_dir, "flow_stats.json"), "w", encoding="utf-8"
    ) as jf:
        json.dump(
            {
                "cases": flow_rows,
                "num_cases": len(flow_rows),
                "total_s": total_t,
            },
            jf,
            indent=1,
        )

    meta = {
        "model_id": m_id,
        "adapter": ad_name,
        "spacing": global_cfg.get("spacing"),
        "num_labels": num_lbl,
        "atlas": atlas,
        "test_glob": test_glob,
        "n_cases": len(flow_rows),
        "jac_impl": global_cfg.get("jac_impl", "vxm_grad"),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as jf:
        json.dump(meta, jf, indent=1)

    # Legacy-compatible line (first row of Results-style CSV) for quick compare
    legacy_dice = ",".join([label_dict[i] for i in range(num_lbl)])
    with open(
        os.path.join(out_dir, "README_compat.txt"), "w", encoding="utf-8"
    ) as rf:
        rf.write(
            f"First column names align with IXI/Results: Unknown,{legacy_dice},non_jec\n"
        )
    return out_dir
