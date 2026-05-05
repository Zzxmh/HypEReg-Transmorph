from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from IXI.analysis_comprehensive.stats import HIGHER_IS_BETTER, _bh_fdr


DEFAULT_MODELS = [
    "transmorph_her",
    "transmorph_original",
    "transmorphbayes",
    "voxelmorph_1",
    "cyclemorph",
    "midir",
    "cotr",
    "nnformer",
    "pvt",
    "syn",
]

DEFAULT_METRICS = ["dice_mean", "HD95_mean", "ASSD_mean", "non_jec", "SDlogJ"]


def _load_per_case(eval_root: str, model: str) -> pd.DataFrame:
    p = os.path.join(eval_root, model, "per_case.csv")
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if "pkl" in df.columns:
        df = df.sort_values("pkl").reset_index(drop=True)
    return df


def _common(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "pkl" not in a.columns or "pkl" not in b.columns:
        n = min(len(a), len(b))
        return a.iloc[:n].reset_index(drop=True), b.iloc[:n].reset_index(drop=True)
    inter = sorted(set(a["pkl"]).intersection(set(b["pkl"])))
    aa = a[a["pkl"].isin(inter)].sort_values("pkl").reset_index(drop=True)
    bb = b[b["pkl"].isin(inter)].sort_values("pkl").reset_index(drop=True)
    return aa, bb


def _paired_effect_signed(diff_aligned: np.ndarray) -> float:
    # Matched-pairs rank-biserial correlation in [-1, 1].
    nz = diff_aligned[np.isfinite(diff_aligned) & (diff_aligned != 0)]
    if nz.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nz), method="average")
    w_pos = float(np.sum(ranks[nz > 0]))
    w_neg = float(np.sum(ranks[nz < 0]))
    den = w_pos + w_neg
    if den == 0:
        return 0.0
    return (w_pos - w_neg) / den


def run(eval_root: str, out_csv: str, ref_model: str, metrics: Iterable[str], models: Iterable[str]) -> None:
    models = [m for m in models if m != ref_model]
    ref_df = _load_per_case(eval_root, ref_model)
    rows: List[Dict[str, object]] = []
    pvals: List[float] = []

    for baseline in models:
        try:
            base_df = _load_per_case(eval_root, baseline)
        except FileNotFoundError:
            continue
        ref_c, base_c = _common(ref_df, base_df)
        for metric in metrics:
            if metric not in ref_c.columns or metric not in base_c.columns:
                continue
            a = pd.to_numeric(ref_c[metric], errors="coerce").to_numpy(dtype=np.float64)
            b = pd.to_numeric(base_c[metric], errors="coerce").to_numpy(dtype=np.float64)
            mask = np.isfinite(a) & np.isfinite(b)
            if int(mask.sum()) < 3:
                continue
            a = a[mask]
            b = b[mask]
            # Raw paired difference: HypEReg - baseline.
            d_raw = a - b
            # Aligned positive means "HypEReg better".
            if metric in HIGHER_IS_BETTER:
                d_aligned = d_raw
            else:
                d_aligned = -d_raw
            try:
                p_raw = float(wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue)
            except Exception:
                p_raw = float("nan")
            if np.isfinite(p_raw):
                pvals.append(p_raw)
            rows.append(
                {
                    "metric": metric,
                    "direction": "higher_better" if metric in HIGHER_IS_BETTER else "lower_better",
                    "baseline": baseline,
                    "n": int(mask.sum()),
                    "her_mean": float(np.mean(a)),
                    "her_std": float(np.std(a, ddof=0)),
                    "baseline_mean": float(np.mean(b)),
                    "baseline_std": float(np.std(b, ddof=0)),
                    "median_paired_diff_raw": float(np.median(d_raw)),
                    "median_paired_diff_aligned": float(np.median(d_aligned)),
                    "effect_signed": float(_paired_effect_signed(d_aligned)),
                    "p_raw": p_raw,
                }
            )

    qvals = _bh_fdr(pvals)
    qi = 0
    for r in rows:
        if np.isfinite(float(r["p_raw"])):
            q = qvals[qi]
            qi += 1
        else:
            q = float("nan")
        r["q_bh_fdr"] = float(q)
        r["significant_q<0.05"] = bool(np.isfinite(q) and q < 0.05)

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["metric", "baseline"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote inferential stats table: {out_csv}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export paired inferential table with p/q/effect size.")
    ap.add_argument("--eval_root", type=str, default=os.path.join("IXI", "Eval_Results"))
    ap.add_argument(
        "--out_csv",
        type=str,
        default=os.path.join("IXI", "Results", "comprehensive", "sig_matrix_extended.csv"),
    )
    ap.add_argument("--ref_model", type=str, default="transmorph_her")
    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    eval_root = args.eval_root if os.path.isabs(args.eval_root) else os.path.join(repo_root, args.eval_root)
    out_csv = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(repo_root, args.out_csv)

    run(
        eval_root=eval_root,
        out_csv=out_csv,
        ref_model=args.ref_model,
        metrics=DEFAULT_METRICS,
        models=DEFAULT_MODELS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

