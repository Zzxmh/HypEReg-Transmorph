from __future__ import annotations

import argparse
import json
import os

import pandas as pd


def _mean_metric(csv_path: str, out_name: str) -> pd.Series:
    df = pd.read_csv(csv_path).set_index("subject")
    metric_cols = list(df.columns)
    return df[metric_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).rename(out_name)


def convert(legacy_dir: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    dice_mean = _mean_metric(os.path.join(legacy_dir, "dice.csv"), "dice_mean")
    jaccard_mean = _mean_metric(os.path.join(legacy_dir, "jaccard.csv"), "jaccard_mean")
    hd95_mean = _mean_metric(os.path.join(legacy_dir, "hd95.csv"), "HD95_mean")
    assd_mean = _mean_metric(os.path.join(legacy_dir, "assd.csv"), "ASSD_mean")

    intensity = pd.read_csv(os.path.join(legacy_dir, "intensity.csv")).set_index("subject")
    jac = pd.read_csv(os.path.join(legacy_dir, "jacobian.csv")).set_index("subject")
    runtime = pd.read_csv(os.path.join(legacy_dir, "runtime.csv")).set_index("subject")

    out = pd.concat(
        [
            dice_mean,
            jaccard_mean,
            hd95_mean,
            assd_mean,
            intensity[["ssim3d", "nmi", "lncc"]].rename(columns={"ssim3d": "SSIM", "nmi": "NMI", "lncc": "LNCC"}),
            jac[["non_jac_frac", "SDlogJ", "J_p01", "J_p50", "J_p99", "J_min", "J_max", "bending_energy", "mean_abs_div"]].rename(
                columns={"non_jac_frac": "non_jec"}
            ),
            runtime[["runtime_s", "peak_mem_GB"]].rename(columns={"runtime_s": "inference_s", "peak_mem_GB": "peak_mem_gb"}),
        ],
        axis=1,
    ).reset_index(drop=True)
    out["ICE"] = float("nan")
    out["stdy_idx"] = out.index
    out.to_csv(os.path.join(out_dir, "per_case.csv"), index=False)

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": "syn",
                "adapter": "syn_legacy_to_per_case",
                "source_dir": legacy_dir,
                "num_cases": int(len(out)),
            },
            f,
            indent=1,
        )
    with open(os.path.join(out_dir, "README_compat.txt"), "w", encoding="utf-8") as f:
        f.write("Converted from comprehensive_syn_lddmm/SyN_IXI/*.csv into per_case schema for stats.py\n")
    return os.path.join(out_dir, "per_case.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy_dir", default=os.path.join("IXI", "Eval_Results", "comprehensive_syn_lddmm", "SyN_IXI"))
    ap.add_argument("--out_dir", default=os.path.join("IXI", "Eval_Results", "syn"))
    args = ap.parse_args()
    p = convert(args.legacy_dir, args.out_dir)
    print(f"Wrote: {p}")
