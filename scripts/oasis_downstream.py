#!/usr/bin/env python3
"""
OASIS downstream experiment: multi-atlas label fusion (D-1).

Majority voting over N atlas-to-target registrations per test subject.
Each model is loaded, all inference run, results saved, then model freed.
This avoids GPU OOM when loading multiple large checkpoints back-to-back.

Usage:
    python scripts/oasis_downstream.py
    python scripts/oasis_downstream.py --models HypEReg-TransMorph
"""
from __future__ import annotations

import argparse
import gc
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ATLAS_IDS = [50, 80, 150, 220, 300, 380]
TEST_IDS = list(range(438, 458))  # 20 test subjects

ROI_LABELS: Dict[str, List[int]] = {
    "Hippocampus": [14, 33],
    "Lateral_Ventricles": [3, 22],
    "Thalamus": [7, 26],
}

OUT_DIR = REPO_ROOT / "OASIS" / "Eval_Results" / "downstream"

MODEL_REGISTRY = [
    ("HypEReg-TransMorph (ZS)", "transmorph_her_zs_oasis"),
    ("TransMorph (ZS)", "transmorph_zs_oasis"),
    ("TransMorphBayes (ZS)", "transmorphbayes_zs_oasis"),
    ("MIDIR (ZS)", "midir_oasis"),
]


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
    return flow_t[0].cpu().numpy().astype(np.float32)


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


def compute_fused_for_target(
    adapter_id: str,
    atlases: Dict[int, Tuple[np.ndarray, np.ndarray]],
    target_img: np.ndarray,
    device: str,
) -> np.ndarray:
    """Load one model, fuse atlas segmentations for a single target, then free GPU memory.

    Returns int32 label map (same shape as ``target_img``), 0..35 OASIS labels.
    """
    import importlib
    import gc
    import torch

    adapter = importlib.import_module(f"OASIS.adapters.{adapter_id}")
    model, _ = adapter.build_model(device)
    model.eval()
    atlas_ids = sorted(atlases.keys())
    warped_segs: List[np.ndarray] = []
    for aid in atlas_ids:
        aim, atl_seg = atlases[aid]
        flow = get_flow(adapter, model, aim, target_img, device)
        warped_segs.append(warp_seg(atl_seg, flow))
        del flow
        gc.collect()
    fused = majority_vote(warped_segs, n=36)
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return fused


def label_dice(pred: np.ndarray, gt: np.ndarray, n: int = 36) -> np.ndarray:
    d = np.full(n, np.nan)
    for c in range(n):
        p, g = pred == c, gt == c
        denom = p.sum() + g.sum()
        d[c] = 1.0 if denom == 0 else 2.0 * float((p & g).sum()) / float(denom)
    return d


def run_one_model(
    model_name: str,
    adapter_id: str,
    atlases: Dict[int, Tuple[np.ndarray, np.ndarray]],
    targets: Dict[int, Tuple[np.ndarray, np.ndarray]],
    device: str,
) -> List[dict]:
    import importlib
    import torch

    print(f"\n=== {model_name} ===", flush=True)
    adapter = importlib.import_module(f"OASIS.adapters.{adapter_id}")
    try:
        model, _ = adapter.build_model(device)
        model.eval()
    except Exception as e:
        print(f"  [error loading]: {e}", flush=True)
        return []

    atlas_ids = sorted(atlases.keys())
    target_ids = sorted(targets.keys())
    d1_rows: List[dict] = []

    for tid in target_ids:
        tim, tgt_seg = targets[tid]
        warped_segs: List[np.ndarray] = []
        single_dices: List[float] = []

        for aid in atlas_ids:
            aim, atl_seg = atlases[aid]
            t0 = time.time()
            try:
                flow = get_flow(adapter, model, aim, tim, device)
            except Exception as e:
                print(f"  [error {aid}->{tid}]: {e}", flush=True)
                continue
            print(f"  {aid}->{tid} done in {time.time()-t0:.2f}s", flush=True)

            ws = warp_seg(atl_seg, flow)
            warped_segs.append(ws)
            d = label_dice(ws, tgt_seg, n=36)
            single_dices.append(float(np.nanmean(d[1:])))

            del flow
            gc.collect()

        if not warped_segs:
            continue

        fused = majority_vote(warped_segs, n=36)
        fd = label_dice(fused, tgt_seg, n=36)
        fd_mean = float(np.nanmean(fd[1:]))
        sd_mean = float(np.nanmean(single_dices))

        row = {
            "model": model_name,
            "target_id": tid,
            "n_atlases": len(warped_segs),
            "single_dice_mean": sd_mean,
            "fused_dice_mean": fd_mean,
            "delta_dice": fd_mean - sd_mean,
        }
        for roi, lids in ROI_LABELS.items():
            row[f"fused_dice_{roi}"] = float(np.nanmean([fd[l] for l in lids if l < 36]))
        d1_rows.append(row)

        del warped_segs
        gc.collect()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return d1_rows


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
        try:
            targets[tid] = load_target(tid)
        except FileNotFoundError:
            print(f"  [skip] target {tid}")
    print(f"  {len(targets)} targets loaded: {sorted(targets)}")

    print("Loading atlases...")
    atlases = {}
    for aid in ATLAS_IDS:
        try:
            atlases[aid] = load_atlas(aid)
        except FileNotFoundError:
            print(f"  [skip] atlas {aid}")
    print(f"  {len(atlases)} atlases loaded: {sorted(atlases)}")

    registry = MODEL_REGISTRY
    if args.models:
        registry = [(n, a) for n, a in registry if n in args.models or a in args.models]

    all_d1: List[dict] = []
    for model_name, adapter_id in registry:
        d1 = run_one_model(model_name, adapter_id, atlases, targets, device)
        all_d1.extend(d1)
        pd.DataFrame(all_d1).to_csv(OUT_DIR / "d1_per_target.csv", index=False)
        print(f"  Saved intermediate results ({len(all_d1)} D-1 rows).")

    d1_df = pd.DataFrame(all_d1)
    if not d1_df.empty:
        agg = d1_df.groupby("model")[["single_dice_mean", "fused_dice_mean", "delta_dice"]].agg(
            ["mean", "std"]
        ).round(4)
        for roi in ROI_LABELS:
            col = f"fused_dice_{roi}"
            if col in d1_df.columns:
                agg[(col, "mean")] = d1_df.groupby("model")[col].mean().round(4)
                agg[(col, "std")] = d1_df.groupby("model")[col].std().round(4)
        agg.to_csv(OUT_DIR / "d1_summary.csv")
        print("\nD-1 Summary:\n", agg)

    print("\nDone.")


if __name__ == "__main__":
    main()
