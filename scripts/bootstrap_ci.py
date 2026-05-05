#!/usr/bin/env python3
"""
Compute bootstrap 95% confidence intervals for IXI subject-level metrics.

Outputs Table S6-ready summaries for five core models:
- HypEReg-TransMorph
- TransMorph
- TransMorphBayes
- MIDIR
- SyN (ANTs)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


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


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    grouped_dice_csv: Path
    per_case_csv: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def model_specs(root: Path) -> List[ModelSpec]:
    return [
        ModelSpec(
            display_name="HypEReg-TransMorph",
            grouped_dice_csv=root / "IXI" / "Results" / "TransMorph_HER_IXI.csv",
            per_case_csv=root / "IXI" / "Eval_Results" / "transmorph_her" / "per_case.csv",
        ),
        ModelSpec(
            display_name="TransMorph",
            grouped_dice_csv=root / "IXI" / "Results" / "TransMorph_ncc_1_diffusion_1.csv",
            per_case_csv=root / "IXI" / "Eval_Results" / "transmorph_original" / "per_case.csv",
        ),
        ModelSpec(
            display_name="TransMorphBayes",
            grouped_dice_csv=root / "IXI" / "Results" / "TransMorphBayes_ncc_1_diffusion_1.csv",
            per_case_csv=root / "IXI" / "Eval_Results" / "transmorphbayes" / "per_case.csv",
        ),
        ModelSpec(
            display_name="MIDIR",
            grouped_dice_csv=root / "IXI" / "Results" / "MIDIR_ncc_1_diffusion_1.csv",
            per_case_csv=root / "IXI" / "Eval_Results" / "midir" / "per_case.csv",
        ),
        ModelSpec(
            display_name="SyN (ANTs)",
            grouped_dice_csv=root / "IXI" / "Results" / "ants_IXI.csv",
            per_case_csv=root / "IXI" / "Eval_Results" / "syn" / "per_case.csv",
        ),
    ]


def parse_legacy_results_rows(csv_path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            if not line:
                continue
            # Legacy exports are usually one comma-joined field per line.
            if len(line) == 1:
                rows.append(line[0].split(","))
            else:
                rows.append(line)
    return rows


def grouped_dice_from_legacy_csv(csv_path: Path) -> np.ndarray:
    rows = parse_legacy_results_rows(csv_path)
    if len(rows) < 3:
        raise ValueError(f"Unexpected legacy CSV format: {csv_path}")

    header = rows[1]
    group_indices: Dict[str, List[int]] = {}
    for stct in OUTSTRUCT:
        idxs = [i for i, name in enumerate(header) if stct in name]
        if not idxs:
            raise ValueError(f"Structure '{stct}' not found in {csv_path.name}")
        group_indices[stct] = idxs

    per_subject_grouped: List[float] = []
    for parts in rows[2:]:
        if len(parts) < 2 or parts[1] == "":
            continue
        group_means: List[float] = []
        for stct in OUTSTRUCT:
            vals = [float(parts[i]) for i in group_indices[stct]]
            group_means.append(float(np.mean(vals)))
        per_subject_grouped.append(float(np.mean(group_means)))

    return np.asarray(per_subject_grouped, dtype=np.float64)


def metric_from_per_case_csv(csv_path: Path, metric: str) -> np.ndarray:
    df = pd.read_csv(csv_path)

    if "stdy_idx" in df.columns:
        # Handle occasional duplicate rows by collapsing to one row per subject.
        df["stdy_idx"] = pd.to_numeric(df["stdy_idx"], errors="coerce")
        df = (
            df.dropna(subset=["stdy_idx"])
            .sort_values("stdy_idx")
            .groupby("stdy_idx", as_index=False)
            .mean(numeric_only=True)
        )

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in {csv_path}")

    values = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy(dtype=np.float64)
    return values


def bootstrap_ci(values: np.ndarray, n_boot: int, ci: float, seed: int) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = values.shape[0]
    samples = rng.integers(0, n, size=(n_boot, n))
    means = values[samples].mean(axis=1)
    alpha = (100.0 - ci) / 2.0
    lo, hi = np.percentile(means, [alpha, 100.0 - alpha])
    return float(values.mean()), float(lo), float(hi)


def fmt_ci(mean: float, lo: float, hi: float, sci: bool = False) -> str:
    if not np.isfinite(mean):
        return "nan [nan, nan]"
    if sci:
        return f"{mean:.3e} [{lo:.3e}, {hi:.3e}]"
    return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap 95% CI for IXI core metrics (Table S6).")
    parser.add_argument("--n-boot", type=int, default=10_000, help="Bootstrap resample count.")
    parser.add_argument("--ci", type=float, default=95.0, help="Confidence level percentage.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for bootstrap resampling.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("IXI/Results/comprehensive/bootstrap_ci_table_s6.csv"),
        help="Output CSV path (relative to repository root unless absolute).",
    )
    args = parser.parse_args()

    root = repo_root()
    out_path = args.output if args.output.is_absolute() else (root / args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric_map = {
        "non_jec": "non_jec",
        "SDlogJ": "SDlogJ",
        "HD95": "HD95_mean",
        "ASSD": "ASSD_mean",
    }

    rows = []
    for spec in model_specs(root):
        dice_values = grouped_dice_from_legacy_csv(spec.grouped_dice_csv)
        non_jec_values = metric_from_per_case_csv(spec.per_case_csv, metric_map["non_jec"])
        sdlogj_values = metric_from_per_case_csv(spec.per_case_csv, metric_map["SDlogJ"])
        hd95_values = metric_from_per_case_csv(spec.per_case_csv, metric_map["HD95"])
        assd_values = metric_from_per_case_csv(spec.per_case_csv, metric_map["ASSD"])

        dice_mean, dice_lo, dice_hi = bootstrap_ci(dice_values, args.n_boot, args.ci, args.seed)
        non_mean, non_lo, non_hi = bootstrap_ci(non_jec_values, args.n_boot, args.ci, args.seed)
        sd_mean, sd_lo, sd_hi = bootstrap_ci(sdlogj_values, args.n_boot, args.ci, args.seed)
        hd_mean, hd_lo, hd_hi = bootstrap_ci(hd95_values, args.n_boot, args.ci, args.seed)
        as_mean, as_lo, as_hi = bootstrap_ci(assd_values, args.n_boot, args.ci, args.seed)

        rows.append(
            {
                "model": spec.display_name,
                "n_subjects": int(dice_values.shape[0]),
                "dice_mean": dice_mean,
                "dice_ci_lo": dice_lo,
                "dice_ci_hi": dice_hi,
                "dice_mean_95ci": fmt_ci(dice_mean, dice_lo, dice_hi),
                "non_jec_mean": non_mean,
                "non_jec_ci_lo": non_lo,
                "non_jec_ci_hi": non_hi,
                "non_jec_mean_95ci": fmt_ci(non_mean, non_lo, non_hi, sci=True),
                "sdlogj_mean": sd_mean,
                "sdlogj_ci_lo": sd_lo,
                "sdlogj_ci_hi": sd_hi,
                "sdlogj_mean_95ci": fmt_ci(sd_mean, sd_lo, sd_hi),
                "hd95_mean": hd_mean,
                "hd95_ci_lo": hd_lo,
                "hd95_ci_hi": hd_hi,
                "hd95_mean_95ci": fmt_ci(hd_mean, hd_lo, hd_hi),
                "assd_mean": as_mean,
                "assd_ci_lo": as_lo,
                "assd_ci_hi": as_hi,
                "assd_mean_95ci": fmt_ci(as_mean, as_lo, as_hi),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {out_path}")
    print(
        out_df[
            [
                "model",
                "dice_mean_95ci",
                "non_jec_mean_95ci",
                "sdlogj_mean_95ci",
                "hd95_mean_95ci",
                "assd_mean_95ci",
            ]
        ]
    )


if __name__ == "__main__":
    main()
