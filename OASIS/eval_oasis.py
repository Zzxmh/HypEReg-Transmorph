from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Dict, List

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import map_coordinates, zoom

from IXI.metrics_full import (
    hd95_assd_mean_over_labels,
    jacobian_stats,
    jaccard_mean_over_labels,
    nmi_3d,
    per_label_dice,
    ssim3d,
    volume_similarity,
)


def _abs(repo_root: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(repo_root, p))


def _ncc_global_numpy(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + eps
    return float((a * b).sum() / denom)


def _bending_energy(flow_czyx: np.ndarray) -> float:
    ux, uy, uz = flow_czyx
    uxx = np.gradient(np.gradient(ux, axis=0), axis=0)
    uyy = np.gradient(np.gradient(ux, axis=1), axis=1)
    uzz = np.gradient(np.gradient(ux, axis=2), axis=2)
    vxx = np.gradient(np.gradient(uy, axis=0), axis=0)
    vyy = np.gradient(np.gradient(uy, axis=1), axis=1)
    vzz = np.gradient(np.gradient(uy, axis=2), axis=2)
    wxx = np.gradient(np.gradient(uz, axis=0), axis=0)
    wyy = np.gradient(np.gradient(uz, axis=1), axis=1)
    wzz = np.gradient(np.gradient(uz, axis=2), axis=2)
    be = (
        uxx**2 + uyy**2 + uzz**2
        + vxx**2 + vyy**2 + vzz**2
        + wxx**2 + wyy**2 + wzz**2
    )
    return float(np.mean(be))


def _mean_abs_div(flow_czyx: np.ndarray) -> float:
    div = (
        np.gradient(flow_czyx[0], axis=0)
        + np.gradient(flow_czyx[1], axis=1)
        + np.gradient(flow_czyx[2], axis=2)
    )
    return float(np.mean(np.abs(div)))


def _load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(cfg_path: str, model_ids: List[str] | None = None) -> None:
    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    cfg = _load_cfg(cfg_path)
    oasis_root = _abs(repo_root, cfg["oasis_root"])
    test_nii_dir = os.path.join(oasis_root, "Test_nii")
    pairs_csv = _abs(repo_root, cfg["pairs_csv"])
    results_root = _abs(repo_root, cfg["results_dir"])
    os.makedirs(results_root, exist_ok=True)

    pairs = pd.read_csv(pairs_csv)
    num_labels = int(cfg.get("num_labels", 36))
    spacing = cfg.get("spacing", [1.0, 1.0, 1.0])
    ssim_win = int(cfg.get("ssim_win", 11))
    nmi_bins = int(cfg.get("nmi_bins", 64))
    n_bins = int(cfg.get("jacobian_hist_bins", 100))

    models = cfg.get("models", [])
    if model_ids:
        models = [m for m in models if m["id"] in set(model_ids)]

    for model in models:
        model_id = model["id"]
        pred_dir = _abs(
            repo_root,
            model.get(
                "pred_dir",
                f"OASIS/data/Submit/submission/{model_id}/task_03",
            ),
        )
        out_dir = os.path.join(results_root, model_id)
        os.makedirs(out_dir, exist_ok=True)

        per_case_path = os.path.join(out_dir, "per_case.csv")
        flow_rows = []
        t0 = time.time()
        with open(per_case_path, "w", newline="", encoding="utf-8") as fcsv:
            writer = None
            for idx, row in pairs.iterrows():
                fixed = int(row["fixed"])
                moving = int(row["moving"])
                case_name = f"{fixed:04d}_{moving:04d}"
                disp_path = os.path.join(pred_dir, f"disp_{case_name}.npz")
                if not os.path.isfile(disp_path):
                    raise FileNotFoundError(f"Missing displacement: {disp_path}")

                fixed_img = nib.load(os.path.join(test_nii_dir, f"img{fixed:04d}.nii.gz")).get_fdata()
                moving_img = nib.load(os.path.join(test_nii_dir, f"img{moving:04d}.nii.gz")).get_fdata()
                fixed_seg = nib.load(os.path.join(test_nii_dir, f"seg{fixed:04d}.nii.gz")).get_fdata()
                moving_seg = nib.load(os.path.join(test_nii_dir, f"seg{moving:04d}.nii.gz")).get_fdata()

                flow = np.load(disp_path)["arr_0"].astype(np.float32)
                if flow.shape[1:] != fixed_img.shape:
                    flow = np.array([zoom(flow[i], 2.0, order=2) for i in range(3)], dtype=np.float32)

                D, H, W = fixed_img.shape
                identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
                coords = np.array(identity) + flow
                warped_img = map_coordinates(moving_img, coords, order=1)
                warped_seg = map_coordinates(moving_seg, coords, order=0)

                dices, _ = per_label_dice(warped_seg.astype(int), fixed_seg.astype(int), num_classes=num_labels)
                vs = volume_similarity(warped_seg.astype(int), fixed_seg.astype(int), num_classes=num_labels)
                jacc = jaccard_mean_over_labels(warped_seg.astype(int), fixed_seg.astype(int), num_classes=num_labels)
                d_mean = float(np.nanmean(dices))
                hd95, assd = hd95_assd_mean_over_labels(
                    warped_seg.astype(int),
                    fixed_seg.astype(int),
                    num_classes=num_labels,
                    spacing=spacing,
                )
                ssim = ssim3d(warped_img, fixed_img, data_range=1.0, win_size=ssim_win)
                nmi = nmi_3d(fixed_img, warped_img, bins=nmi_bins)
                lncc = _ncc_global_numpy(fixed_img, warped_img)
                jstat = jacobian_stats(flow, n_bins=n_bins)
                bending = _bending_energy(flow)
                mean_abs_div = _mean_abs_div(flow)

                if writer is None:
                    fields = [f"dice_{i}" for i in range(num_labels)]
                    fields += [f"vs_{i}" for i in range(num_labels)]
                    fields += [
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
                        "bending_energy",
                        "mean_abs_div",
                        "inference_s",
                        "peak_mem_gb",
                        "ICE",
                        "pkl",
                        "stdy_idx",
                    ]
                    writer = csv.DictWriter(fcsv, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader()

                out = {f"dice_{i}": dices[i] for i in range(num_labels)}
                out.update({f"vs_{i}": vs[i] for i in range(num_labels)})
                out.update(
                    {
                        "jaccard_mean": jacc,
                        "dice_mean": d_mean,
                        "HD95_mean": hd95,
                        "ASSD_mean": assd,
                        "SSIM": ssim,
                        "NMI": nmi,
                        "LNCC": lncc,
                        "non_jec": jstat["non_jec"],
                        "SDlogJ": jstat["SDlogJ"],
                        "J_p01": jstat["J_p01"],
                        "J_p50": jstat["J_p50"],
                        "J_p99": jstat["J_p99"],
                        "J_min": jstat["J_min"],
                        "J_max": jstat["J_max"],
                        "bending_energy": bending,
                        "mean_abs_div": mean_abs_div,
                        "inference_s": np.nan,
                        "peak_mem_gb": np.nan,
                        "ICE": np.nan,
                        "pkl": f"p_{case_name}",
                        "stdy_idx": idx,
                    }
                )
                writer.writerow(out)
                flow_rows.append(
                    {
                        "case": case_name,
                        "hist_counts": jstat["hist_counts"],
                        "hist_edges": jstat["hist_edges"],
                        "non_jec": jstat["non_jec"],
                        "bending_energy": bending,
                        "mean_abs_div": mean_abs_div,
                    }
                )

        with open(os.path.join(out_dir, "flow_stats.json"), "w", encoding="utf-8") as f:
            json.dump({"cases": flow_rows, "num_cases": len(flow_rows), "total_s": time.time() - t0}, f, indent=1)
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_id": model_id,
                    "supervision": model.get("supervision", "unknown"),
                    "pred_dir": pred_dir,
                    "pairs_csv": pairs_csv,
                    "num_cases": len(flow_rows),
                    "num_labels": num_labels,
                },
                f,
                indent=1,
            )
        print(f"[OASIS-Eval] {model_id}: {len(flow_rows)} cases -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OASIS predictions with IXI-compatible metric panel")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "eval_configs.yaml"))
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()
    run(args.config, args.models)
