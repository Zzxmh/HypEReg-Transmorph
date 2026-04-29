# -*- coding: utf-8 -*-
"""
Map IXI/Results/*.csv (Dice + non_jec) to the same column schema as Eval_Results/**/per_case.csv
(missing full-metric fields left empty or NaN).
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from typing import List, Optional

import numpy as np

IXI_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(IXI_DIR)

DEFAULT_PATTERNS = [
    "CoTr_ncc_1_diffusion_1",
    "PVT_ncc_1_diffusion_1",
    "ViTVNet_ncc_1_diffusion_1",
    "nnFormer_ncc_1_diffusion_1",
    "TransMorphDiff",
    "TransMorphBspline_ncc_1_diffusion_1",
    "TransMorphBayes_ncc_1_diffusion_1",
    "TransMorph_ncc_1_diffusion_1",
    "TransMorph_HER_IXI",
]


def per_case_fieldnames(num_labels: int = 46) -> List[str]:
    h = [f"dice_{i}" for i in range(num_labels)]
    h += [f"vs_{i}" for i in range(num_labels)]
    h += [
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
    return h


def _parse_one_legacy_csv(
    path: str, num_labels: int = 46
) -> List[dict]:
    """
    Infers rows from IXI/Results/* style: line0 model name, line1 header, line2+ data.
    Data rows: p_k,dice0,...,dice45,non_jec
    """
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        content = f.read()
    # normalize newlines; split
    lines = [ln for ln in content.splitlines() if ln.strip() != ""]
    if len(lines) < 2:
        return []
    # Second line is often header; data starts line index 2 (0-based) if first is title
    start = 0
    if not lines[0].lower().startswith("p_"):
        start = 1
    if start + 1 >= len(lines):
        return []
    header_line = lines[start]
    # skip header "line" and take data
    data_lines = lines[start + 1 :]
    out = []
    for di, line in enumerate(data_lines):
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0].startswith("p_"):
            continue
        dices: List[Optional[float]] = []
        for i in range(num_labels):
            idx = 1 + i
            if idx < len(parts):
                try:
                    dices.append(float(parts[idx]))
                except ValueError:
                    dices.append(float("nan"))
            else:
                dices.append(float("nan"))
        # non_jec last
        try:
            nj = float(parts[-1]) if len(parts) > 1 else float("nan")
        except ValueError:
            nj = float("nan")
        row: dict = {f"dice_{i}": dices[i] for i in range(num_labels)}
        for i in range(num_labels):
            row[f"vs_{i}"] = np.nan
        row.update(
            {
                "jaccard_mean": np.nan,
                "dice_mean": float(
                    np.nanmean([d for d in dices if np.isfinite(d)])
                )
                if any(np.isfinite(d) for d in dices)
                else np.nan,
                "HD95_mean": np.nan,
                "ASSD_mean": np.nan,
                "SSIM": np.nan,
                "NMI": np.nan,
                "LNCC": np.nan,
                "non_jec": nj,
                "SDlogJ": np.nan,
                "J_p01": np.nan,
                "J_p50": np.nan,
                "J_p99": np.nan,
                "J_min": np.nan,
                "J_max": np.nan,
                "inference_s": np.nan,
                "peak_mem_gb": np.nan,
                "ICE": np.nan,
                "pkl": f"legacy_row_{di}",
                "stdy_idx": di,
            }
        )
        out.append(row)
    return out


def run(
    results_dir: Optional[str] = None,
    out_subdir: str = "legacy_from_results",
) -> str:
    rdir = results_dir or os.path.join(IXI_DIR, "Results")
    out_base = os.path.join(IXI_DIR, "Eval_Results", out_subdir)
    os.makedirs(out_base, exist_ok=True)
    for name in DEFAULT_PATTERNS:
        p = os.path.join(rdir, f"{name}.csv")
        if not os.path.isfile(p):
            print(f"Skip missing: {p}", file=sys.stderr)
            continue
        rows = _parse_one_legacy_csv(p)
        if not rows:
            print(f"No rows: {p}", file=sys.stderr)
            continue
        fn = per_case_fieldnames()
        out_path = os.path.join(out_base, f"{name}_per_case.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("Wrote", out_path, "n=", len(rows))
    return out_base


if __name__ == "__main__":
    run()
