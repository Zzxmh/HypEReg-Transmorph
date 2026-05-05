#!/usr/bin/env python3
"""
OASIS Downstream Experiments D-1 and D-2.

D-1  Multi-Atlas Label Fusion (Majority Voting, N=5 atlases, 20 targets)
D-2  ROI Volumetric Reliability (Jacobian Integration + ICC)

Each model is loaded, all inference run, results saved, then model freed.
This avoids GPU OOM when loading multiple large checkpoints back-to-back.

Usage:
    python scripts/oasis_downstream.py
    python scripts/oasis_downstream.py --models HypEReg-TransMorph
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ATLAS_IDS = [50, 80, 150, 220, 300, 380]
TEST_IDS  = list(range(438, 458))          # 20 test subjects

ROI_LABELS: Dict[str, List[int]] = {
    "Hippocampus":       [14, 33],
    "Lateral_Ventricles":[3,  22],
    "Thalamus":          [7,  26],
}

OUT_DIR = REPO_ROOT / "OASIS" / "Eval_Results" / "downstream"

MODEL_REGISTRY = [
    ("HypEReg-TransMorph (ZS)", "transmorph_her_zs_oasis"),
    ("TransMorph (ZS)",          "transmorph_zs_oasis"),
    ("TransMorphBayes (ZS)",     "transmorphbayes_zs_oasis"),
]


# ─── Data loading ────────────────────────────────────────────────────────────

def load_atlas(aid: int) -> Tuple[np.ndarray, np.ndarray]:
    p = REPO_ROOT / "OASIS" / "data" / "All" / f"p_{aid:04d}.pkl"
    with open(p, "rb") as f:
        data = pickle.load(f)
    return data[0].astype(np.float32), data[1].astype(np.int32)


def load_target(tid: int) -> Tuple[np.ndarray, np.ndarray]:
    d = REPO_ROOT / "OASIS" / "data" / "Test_nii"
    img = nib.load(str(d / f"img{tid:04d}.nii.gz")).get_fdata().astype(np.float32)
    seg = nib.load(str(d / f"seg{tid:04d}.nii.gz")).get_fdata().astype(np.int32)
    return img, seg


# ─── Model inference ─────────────────────────────────────────────────────────

def get_flow(adapter_mod, model, atlas_img, target_img, device) -> np.ndarray:
    import torch
    x = torch.from_numpy(atlas_img[None, None]).to(device)
    y = torch.from_numpy(target_img[None, None]).to(device)
    with torch.no_grad():
        out = adapter_mod.forward(model, x, y)
    if isinstance(out, (list, tuple)):
        flow_t = next((t for t in out if t.ndim == 5 and t.shape[1] == 3), out[-1])
    else:
        flow_t = out
    return flow_t[0].cpu().numpy().astype(np.float32)   # (3,D,H,W)


# ─── Label fusion ────────────────────────────────────────────────────────────

def warp_seg(seg: np.ndarray, flow: np.ndarray) -> np.ndarray:
    D, H, W = seg.shape
    g = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    coords = np.array(g, dtype=np.float32) + flow
    return map_coordinates(seg.astype(np.float32), coords, order=0, mode="nearest").astype(np.int32)


def majority_vote(segs: List[np.ndarray], n: int = 36) -> np.ndarray:
    votes = np.zeros((n,) + segs[0].shape, dtype=np.int16)
    for s in segs:
        for c in range(n):
            votes[c] += (s == c).astype(np.int16)
    return votes.argmax(axis=0).astype(np.int32)


def label_dice(pred: np.ndarray, gt: np.ndarray, n: int = 36) -> np.ndarray:
    d = np.full(n, np.nan)
    for c in range(n):
        p, g = pred == c, gt == c
        denom = p.sum() + g.sum()
        d[c] = 1.0 if denom == 0 else 2.0 * float((p & g).sum()) / float(denom)
    return d


# ─── Jacobian ────────────────────────────────────────────────────────────────

def jac_det(flow: np.ndarray) -> np.ndarray:
    """Jacobian determinant of phi=Id+flow. flow: (3,D,H,W) -> det: (D-1,H-1,W-1)."""
    u = flow  # (3,D,H,W)
    # Forward finite differences on interior lattice (D-1, H-1, W-1)
    du_dx = u[0, 1:, :-1, :-1] - u[0, :-1, :-1, :-1]
    du_dy = u[0, :-1, 1:, :-1] - u[0, :-1, :-1, :-1]
    du_dz = u[0, :-1, :-1, 1:] - u[0, :-1, :-1, :-1]
    dv_dx = u[1, 1:, :-1, :-1] - u[1, :-1, :-1, :-1]
    dv_dy = u[1, :-1, 1:, :-1] - u[1, :-1, :-1, :-1]
    dv_dz = u[1, :-1, :-1, 1:] - u[1, :-1, :-1, :-1]
    dw_dx = u[2, 1:, :-1, :-1] - u[2, :-1, :-1, :-1]
    dw_dy = u[2, :-1, 1:, :-1] - u[2, :-1, :-1, :-1]
    dw_dz = u[2, :-1, :-1, 1:] - u[2, :-1, :-1, :-1]
    J11, J12, J13 = 1 + du_dx, du_dy, du_dz
    J21, J22, J23 = dv_dx, 1 + dv_dy, dv_dz
    J31, J32, J33 = dw_dx, dw_dy, 1 + dw_dz
    return J11*(J22*J33 - J23*J32) - J12*(J21*J33 - J23*J31) + J13*(J21*J32 - J22*J31)


def roi_jac_vol(det: np.ndarray, atlas_seg: np.ndarray, lids: List[int]) -> float:
    seg = atlas_seg[:-1, :-1, :-1]
    m = np.zeros(seg.shape, dtype=bool)
    for l in lids: m |= (seg == l)
    return float(det[m].sum()) if m.sum() > 0 else np.nan


def ref_vol(seg: np.ndarray, lids: List[int]) -> float:
    m = np.zeros(seg.shape, dtype=bool)
    for l in lids: m |= (seg == l)
    return float(m.sum())


def icc31(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 3: return np.nan
    gm = (x + y).mean() / 2
    msr = n * ((x.mean()-gm)**2 + (y.mean()-gm)**2)
    mse = (((x-x.mean())**2 + (y-y.mean())**2).sum()) / (2*(n-1))
    return float((msr-mse)/(msr+mse)) if (msr+mse) > 1e-12 else 1.0


# ─── Main experiment ─────────────────────────────────────────────────────────

def run_one_model(
    model_name: str,
    adapter_id: str,
    atlases: Dict[int, Tuple[np.ndarray, np.ndarray]],
    targets: Dict[int, Tuple[np.ndarray, np.ndarray]],
    device: str,
) -> Tuple[List[dict], List[dict]]:
    import importlib
    import torch

    print(f"\n=== {model_name} ===", flush=True)
    adapter = importlib.import_module(f"OASIS.adapters.{adapter_id}")
    try:
        model, _ = adapter.build_model(device)
        model.eval()
    except Exception as e:
        print(f"  [error loading]: {e}", flush=True)
        return [], []

    atlas_ids = sorted(atlases.keys())
    target_ids = sorted(targets.keys())

    # Inference: all (atlas, target) pairs
    flows: Dict[Tuple[int,int], np.ndarray] = {}
    for aid in atlas_ids:
        aim, _ = atlases[aid]
        for tid in target_ids:
            tim, _ = targets[tid]
            t0 = time.time()
            try:
                f = get_flow(adapter, model, aim, tim, device)
                flows[(aid, tid)] = f
                print(f"  {aid}->{tid} done in {time.time()-t0:.2f}s", flush=True)
            except Exception as e:
                print(f"  [error {aid}->{tid}]: {e}", flush=True)

    # Free GPU memory immediately
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # D-1: fusion Dice + D-2: volumetric reliability
    print(f"  Computing D-1/D-2 for {model_name}...", flush=True)
    d1_rows, d2_rows = [], []
    for tid in target_ids:
        _, tgt_seg = targets[tid]
        warped_segs, single_dices_mean, jac_vols = [], [], {r:[] for r in ROI_LABELS}

        for aid in atlas_ids:
            if (aid, tid) not in flows: continue
            _, atl_seg = atlases[aid]
            flow = flows[(aid, tid)]
            ws = warp_seg(atl_seg, flow)
            warped_segs.append(ws)
            d = label_dice(ws, tgt_seg, n=36)
            single_dices_mean.append(float(np.nanmean(d[1:])))

            det = jac_det(flow)
            for roi, lids in ROI_LABELS.items():
                jac_vols[roi].append(roi_jac_vol(det, atl_seg, lids))

        if not warped_segs: continue

        fused = majority_vote(warped_segs, n=36)
        fd = label_dice(fused, tgt_seg, n=36)
        fd_mean = float(np.nanmean(fd[1:]))
        sd_mean = float(np.nanmean(single_dices_mean))

        row = {"model":model_name,"target_id":tid,"n_atlases":len(warped_segs),
               "single_dice_mean":sd_mean,"fused_dice_mean":fd_mean,
               "delta_dice":fd_mean-sd_mean}
        for roi, lids in ROI_LABELS.items():
            row[f"fused_dice_{roi}"] = float(np.nanmean([fd[l] for l in lids if l<36]))
        d1_rows.append(row)

        for roi, lids in ROI_LABELS.items():
            rv = ref_vol(tgt_seg, lids)
            pv = ref_vol(fused, lids)
            jv = float(np.nanmean(jac_vols[roi])) if jac_vols[roi] else np.nan
            d2_rows.append({
                "model":model_name, "target_id":tid, "roi":roi,
                "ref_vol_vox":rv, "prop_vol_vox":pv, "jac_vol_vox":jv,
                "are_prop_pct": abs(pv-rv)/rv*100 if rv>0 else np.nan,
                "are_jac_pct":  abs(jv-rv)/rv*100 if (rv>0 and np.isfinite(jv)) else np.nan,
            })

    return d1_rows, d2_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--models", nargs="*", help="Subset of model display names")
    ap.add_argument("--atlas-ids", nargs="+", type=int, default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.atlas_ids:
        ATLAS_IDS[:] = args.atlas_ids

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading targets...")
    targets = {}
    for tid in TEST_IDS:
        try: targets[tid] = load_target(tid)
        except FileNotFoundError: print(f"  [skip] target {tid}")
    print(f"  {len(targets)} targets loaded: {sorted(targets)}")

    print("Loading atlases...")
    atlases = {}
    for aid in ATLAS_IDS:
        try: atlases[aid] = load_atlas(aid)
        except FileNotFoundError: print(f"  [skip] atlas {aid}")
    print(f"  {len(atlases)} atlases loaded: {sorted(atlases)}")

    registry = MODEL_REGISTRY
    if args.models:
        registry = [(n, a) for n, a in registry if n in args.models or a in args.models]

    all_d1, all_d2 = [], []
    for model_name, adapter_id in registry:
        d1, d2 = run_one_model(model_name, adapter_id, atlases, targets, device)
        all_d1.extend(d1); all_d2.extend(d2)
        # Save after each model (in case later model crashes)
        pd.DataFrame(all_d1).to_csv(OUT_DIR/"d1_per_target.csv", index=False)
        pd.DataFrame(all_d2).to_csv(OUT_DIR/"d2_per_target_roi.csv", index=False)
        print(f"  Saved intermediate results ({len(all_d1)} D-1 rows).")

    # Aggregate D-1
    d1_df = pd.DataFrame(all_d1)
    if not d1_df.empty:
        agg = d1_df.groupby("model")[["single_dice_mean","fused_dice_mean","delta_dice"]].agg(["mean","std"]).round(4)
        for roi in ROI_LABELS:
            col = f"fused_dice_{roi}"
            if col in d1_df.columns:
                agg[(col,"mean")] = d1_df.groupby("model")[col].mean().round(4)
                agg[(col,"std")]  = d1_df.groupby("model")[col].std().round(4)
        agg.to_csv(OUT_DIR/"d1_summary.csv")
        print("\nD-1 Summary:\n", agg)

    # Aggregate D-2 with ICC
    d2_df = pd.DataFrame(all_d2)
    if not d2_df.empty:
        icc_rows = []
        for mn in d2_df["model"].unique():
            for roi in ROI_LABELS:
                s = d2_df[(d2_df["model"]==mn)&(d2_df["roi"]==roi)].dropna(subset=["ref_vol_vox"])
                rv = s["ref_vol_vox"].values
                pv = s["prop_vol_vox"].values
                jv = s["jac_vol_vox"].values
                vj = np.isfinite(jv)
                icc_rows.append({
                    "model":mn, "roi":roi,
                    "are_prop_mean": s["are_prop_pct"].mean(),
                    "are_prop_std":  s["are_prop_pct"].std(),
                    "are_jac_mean":  s["are_jac_pct"].dropna().mean() if vj.sum()>0 else np.nan,
                    "are_jac_std":   s["are_jac_pct"].dropna().std()  if vj.sum()>0 else np.nan,
                    "icc_prop": icc31(rv, pv),
                    "icc_jac":  icc31(rv[vj], jv[vj]) if vj.sum()>=3 else np.nan,
                    "ba_bias_prop": float((pv-rv).mean()),
                    "ba_bias_jac":  float((jv[vj]-rv[vj]).mean()) if vj.sum()>0 else np.nan,
                })
        d2_icc = pd.DataFrame(icc_rows)
        d2_icc.to_csv(OUT_DIR/"d2_icc_summary.csv", index=False)
        print("\nD-2 ICC Summary:\n", d2_icc)

    print("\nDone.")


if __name__ == "__main__":
    main()
