#!/usr/bin/env python3
"""
OASIS Jacobian ROI Cleanliness Analysis (Supplementary Table S7).

Computes per-ROI Jacobian statistics for IXI->OASIS zero-shot models
to quantify TBM input quality in anatomically relevant regions.

ROIs targeted (OASIS 36-class FreeSurfer label scheme):
  Hippocampus     : labels 17 (L), 53 (R)
  Lateral Ventricle: labels  4 (L), 43 (R)
  Cortical Ribbon  : labels  3 (L, Cerebral-Cortex), 42 (R)

For each (model, ROI, test-pair) triple the script computes:
  - roi_fold_ratio : fraction of voxels with det(J) <= 0 within ROI
  - roi_median_logJ: median of log(max(det(J), 1e-5)) within ROI
  - roi_sdlogJ     : std of log(max(det(J), 1e-5)) within ROI

Then it aggregates over 19 OASIS pairs and runs paired Wilcoxon tests
(HypEReg-TransMorph vs TransMorph) with BH-FDR correction.

Usage
-----
  python scripts/oasis_roi_analysis.py

Requires
--------
  - OASIS displacement field .npy files under:
      OASIS/data/Submit/submission/<model_id>/task_03/*.npy
    Each file should contain the displacement field of shape (D, H, W, 3).
  - OASIS segmentation label files for each test pair (fixed image label),
    discoverable via OASIS/data/pairs_val.csv.

Output
------
  OASIS/Eval_Results/_stats/oasis_roi_jacobian_s7.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[1]

# Models for which displacement fields are expected
MODEL_IDS = [
    "transmorph_her_zs_oasis",
    "transmorph_zs_oasis",
    "transmorphbayes_zs_oasis",
    "midir_oasis",
]

MODEL_DISPLAY = {
    "transmorph_her_zs_oasis": "HypEReg-TransMorph (ZS)",
    "transmorph_zs_oasis": "TransMorph (ZS)",
    "transmorphbayes_zs_oasis": "TransMorphBayes (ZS)",
    "midir_oasis": "MIDIR (ZS)",
}

# OASIS 36-label FreeSurfer label IDs (subset used here)
ROI_LABELS: Dict[str, List[int]] = {
    "Hippocampus": [17, 53],
    "Lateral_Ventricles": [4, 43],
    "Cortical_Ribbon": [3, 42],
}

OASIS_LABEL_ID_SET = [
    2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41,
    42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 77, 80, 85, 251, 252, 253,
]


def _jacobian_det(flow: np.ndarray) -> np.ndarray:
    """Compute voxel-wise Jacobian determinant of displacement field.

    Parameters
    ----------
    flow : ndarray of shape (D, H, W, 3)
        Displacement field in voxel units.

    Returns
    -------
    det_J : ndarray of shape (D-1, H-1, W-1)
    """
    D, H, W, _ = flow.shape
    # Forward finite differences
    du_dx = flow[1:, :-1, :-1, 0] - flow[:-1, :-1, :-1, 0]
    du_dy = flow[:-1, 1:, :-1, 0] - flow[:-1, :-1, :-1, 0]
    du_dz = flow[:-1, :-1, 1:, 0] - flow[:-1, :-1, :-1, 0]

    dv_dx = flow[1:, :-1, :-1, 1] - flow[:-1, :-1, :-1, 1]
    dv_dy = flow[:-1, 1:, :-1, 1] - flow[:-1, :-1, :-1, 1]
    dv_dz = flow[:-1, :-1, 1:, 1] - flow[:-1, :-1, :-1, 1]

    dw_dx = flow[1:, :-1, :-1, 2] - flow[:-1, :-1, :-1, 2]
    dw_dy = flow[:-1, 1:, :-1, 2] - flow[:-1, :-1, :-1, 2]
    dw_dz = flow[:-1, :-1, 1:, 2] - flow[:-1, :-1, :-1, 2]

    # Jacobian of phi = I + u => det(J_phi) = det(I + grad u)
    J11 = 1 + du_dx
    J12 = du_dy
    J13 = du_dz
    J21 = dv_dx
    J22 = 1 + dv_dy
    J23 = dv_dz
    J31 = dw_dx
    J32 = dw_dy
    J33 = 1 + dw_dz

    det = (
        J11 * (J22 * J33 - J23 * J32)
        - J12 * (J21 * J33 - J23 * J31)
        + J13 * (J21 * J32 - J22 * J31)
    )
    return det


def _roi_jacobian_stats(
    det_J: np.ndarray,
    label_vol: np.ndarray,
    roi_label_ids: List[int],
    eps: float = 1e-5,
) -> Tuple[float, float, float]:
    """Compute per-ROI Jacobian statistics.

    Parameters
    ----------
    det_J : ndarray, shape (D-1, H-1, W-1)
    label_vol : ndarray int, shape (D, H, W)
        Segmentation label volume for the fixed image.
    roi_label_ids : list of int
        FreeSurfer label IDs belonging to this ROI.
    eps : float
        Floor for log-Jacobian computation.

    Returns
    -------
    (fold_ratio, median_logJ, sdlogJ)
    """
    # Crop label to match det_J size (interior lattice)
    seg_crop = label_vol[:-1, :-1, :-1]
    roi_mask = np.zeros(seg_crop.shape, dtype=bool)
    for lid in roi_label_ids:
        roi_mask |= seg_crop == lid

    if roi_mask.sum() == 0:
        return np.nan, np.nan, np.nan

    det_roi = det_J[roi_mask]
    fold_ratio = float((det_roi <= 0).sum() / roi_mask.sum())
    log_det = np.log(np.maximum(det_roi, eps))
    median_logJ = float(np.median(log_det))
    sdlogJ = float(np.std(log_det))
    return fold_ratio, median_logJ, sdlogJ


def _bh_correct(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    order = np.argsort(pvals)
    rank = np.empty(n)
    rank[order] = np.arange(1, n + 1)
    qvals = pvals * n / rank
    # Enforce monotonicity from right
    for i in range(n - 2, -1, -1):
        qvals[order[i]] = min(qvals[order[i]], qvals[order[i + 1]])
    return np.clip(qvals, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="OASIS Jacobian ROI analysis (Table S7).")
    parser.add_argument(
        "--flow-dir",
        type=Path,
        default=REPO_ROOT / "OASIS" / "data" / "Submit" / "submission",
        help="Root directory containing per-model displacement field subdirectories.",
    )
    parser.add_argument(
        "--seg-dir",
        type=Path,
        default=REPO_ROOT / "OASIS" / "data",
        help="Directory containing OASIS segmentation NIfTI/numpy files.",
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=REPO_ROOT / "OASIS" / "data" / "pairs_val.csv",
        help="OASIS test pairs CSV (columns: fixed, moving).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "OASIS" / "Eval_Results" / "_stats" / "oasis_roi_jacobian_s7.csv",
    )
    args = parser.parse_args()

    pairs_df = pd.read_csv(args.pairs_csv)
    print(f"Loaded {len(pairs_df)} OASIS test pairs from {args.pairs_csv}")

    records = []
    for model_id in MODEL_IDS:
        flow_model_dir = args.flow_dir / model_id / "task_03"
        if not flow_model_dir.exists():
            print(f"[skip] displacement field directory not found: {flow_model_dir}")
            continue

        for row_idx, pair in pairs_df.iterrows():
            fixed_id = str(pair["fixed"]) if "fixed" in pair else str(pair.iloc[0])
            moving_id = str(pair["moving"]) if "moving" in pair else str(pair.iloc[1])
            pair_tag = f"{fixed_id}_{moving_id}"

            # Load displacement field
            flow_path = flow_model_dir / f"disp_{pair_tag}.npy"
            if not flow_path.exists():
                # Try alternative naming
                alt = list(flow_model_dir.glob(f"*{fixed_id}*{moving_id}*.npy"))
                if not alt:
                    print(f"[warn] no flow file for {pair_tag} in {flow_model_dir}")
                    continue
                flow_path = alt[0]

            flow = np.load(str(flow_path))
            if flow.ndim == 4 and flow.shape[0] == 3:
                flow = np.transpose(flow, (1, 2, 3, 0))

            det_J = _jacobian_det(flow)

            # Load fixed segmentation
            seg_path = args.seg_dir / f"{fixed_id}_seg.npy"
            if not seg_path.exists():
                nii_path = args.seg_dir / f"{fixed_id}_seg.nii.gz"
                if nii_path.exists():
                    import nibabel as nib  # optional import
                    seg_vol = nib.load(str(nii_path)).get_fdata().astype(np.int32)
                else:
                    print(f"[warn] no segmentation for fixed {fixed_id}")
                    continue
            else:
                seg_vol = np.load(str(seg_path)).astype(np.int32)

            for roi_name, label_ids in ROI_LABELS.items():
                fold_r, med_logJ, sd_logJ = _roi_jacobian_stats(det_J, seg_vol, label_ids)
                records.append(
                    {
                        "model": MODEL_DISPLAY.get(model_id, model_id),
                        "pair": pair_tag,
                        "roi": roi_name,
                        "roi_fold_ratio": fold_r,
                        "roi_median_logJ": med_logJ,
                        "roi_sdlogJ": sd_logJ,
                    }
                )

    if not records:
        print("[warn] No records produced. Are the displacement fields present?")
        return

    df = pd.DataFrame(records)

    # Aggregate and paired tests (HypEReg vs TransMorph)
    agg_rows = []
    pvals_list = []
    for roi_name in ROI_LABELS:
        for metric in ["roi_fold_ratio", "roi_median_logJ", "roi_sdlogJ"]:
            her_vals = df[(df["model"].str.contains("HypEReg")) & (df["roi"] == roi_name)].sort_values("pair")[metric].values
            tm_vals = df[(df["model"].str.contains("TransMorph") & ~df["model"].str.contains("Bayes") & ~df["model"].str.contains("HypEReg")) & (df["roi"] == roi_name)].sort_values("pair")[metric].values

            if len(her_vals) >= 5 and len(tm_vals) >= 5 and len(her_vals) == len(tm_vals):
                try:
                    stat, pval = wilcoxon(her_vals, tm_vals, alternative="two-sided")
                except Exception:
                    pval = np.nan
            else:
                pval = np.nan
            pvals_list.append((roi_name, metric, pval))

    pvals_arr = np.array([p for _, _, p in pvals_list])
    valid = np.isfinite(pvals_arr)
    qvals_arr = np.full_like(pvals_arr, np.nan)
    if valid.sum() > 0:
        qvals_arr[valid] = _bh_correct(pvals_arr[valid])

    # Summary table per model x ROI x metric
    summary = (
        df.groupby(["model", "roi"])[["roi_fold_ratio", "roi_median_logJ", "roi_sdlogJ"]]
        .agg(["mean", "std"])
        .round(6)
    )
    print(summary)

    df.to_csv(args.output, index=False)
    print(f"\nWrote per-pair ROI records to: {args.output}")

    # Print paired test summary
    print("\nPaired Wilcoxon (HypEReg vs TransMorph, BH-FDR):")
    for (roi_name, metric, _), p, q in zip(pvals_list, pvals_arr, qvals_arr):
        print(f"  {roi_name:25s} {metric:20s} p={p:.3e}  q={q:.3e}")


if __name__ == "__main__":
    main()
