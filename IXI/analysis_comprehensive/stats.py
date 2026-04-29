"""
Paired significance testing for comprehensive IXI evaluations.

Reads IXI/Eval_Results/*/per_case.csv and writes:
  - model_summary.csv
  - sig_matrix.csv   (HER vs each baseline, Wilcoxon + BH-FDR)
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from pandas.errors import EmptyDataError


DEFAULT_METRICS = [
    "dice_mean",
    "jaccard_mean",
    "HD95_mean",
    "ASSD_mean",
    "NMI",
    "LNCC",
    "SSIM",
    "non_jec",
    "SDlogJ",
    "J_p01",
    "J_p99",
    "inference_s",
    "peak_mem_gb",
]

HIGHER_IS_BETTER = {"dice_mean", "jaccard_mean", "NMI", "LNCC", "SSIM", "J_p01"}


@dataclass
class PValRow:
    metric: str
    baseline: str
    n: int
    p_raw: float


def _bh_fdr(pvals: List[float]) -> List[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    sorted_p = np.asarray([pvals[i] for i in order], dtype=np.float64)
    q = np.empty(m, dtype=np.float64)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = float(sorted_p[i] * m / rank)
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(m, dtype=np.float64)
    for i, idx in enumerate(order):
        out[idx] = min(1.0, max(0.0, q[i]))
    return out.tolist()


def _discover(eval_root: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if not os.path.isdir(eval_root):
        return out
    for name in sorted(os.listdir(eval_root)):
        p = os.path.join(eval_root, name, "per_case.csv")
        if os.path.isfile(p):
            try:
                df = pd.read_csv(p)
            except EmptyDataError:
                continue
            if "pkl" in df.columns:
                df = df.sort_values("pkl").reset_index(drop=True)
            out[name] = df
    return out


def _common_subjects(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "pkl" not in a.columns or "pkl" not in b.columns:
        n = min(len(a), len(b))
        return a.iloc[:n].reset_index(drop=True), b.iloc[:n].reset_index(drop=True)
    inter = sorted(set(a["pkl"]).intersection(set(b["pkl"])))
    aa = a[a["pkl"].isin(inter)].sort_values("pkl").reset_index(drop=True)
    bb = b[b["pkl"].isin(inter)].sort_values("pkl").reset_index(drop=True)
    return aa, bb


def _summary_rows(data: Dict[str, pd.DataFrame], metrics: Iterable[str]) -> pd.DataFrame:
    rows = []
    for model, df in data.items():
        row = {"model": model, "n": int(len(df))}
        for m in metrics:
            if m not in df.columns:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan
                continue
            v = pd.to_numeric(df[m], errors="coerce").to_numpy(dtype=np.float64)
            row[f"{m}_mean"] = float(np.nanmean(v))
            row[f"{m}_std"] = float(np.nanstd(v, ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_pvals(
    data: Dict[str, pd.DataFrame],
    ref_model: str,
    metrics: Iterable[str],
) -> List[PValRow]:
    if ref_model not in data:
        raise ValueError(f"Reference model `{ref_model}` not found under Eval_Results.")
    out: List[PValRow] = []
    ref = data[ref_model]
    for model, df in data.items():
        if model == ref_model:
            continue
        a_df, b_df = _common_subjects(ref, df)
        for m in metrics:
            if m not in a_df.columns or m not in b_df.columns:
                continue
            a = pd.to_numeric(a_df[m], errors="coerce").to_numpy(dtype=np.float64)
            b = pd.to_numeric(b_df[m], errors="coerce").to_numpy(dtype=np.float64)
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 3:
                continue
            try:
                res = wilcoxon(a[mask], b[mask], alternative="two-sided", zero_method="wilcox")
                p = float(res.pvalue)
            except Exception:
                p = float("nan")
            out.append(PValRow(metric=m, baseline=model, n=int(mask.sum()), p_raw=p))
    return out


def run(
    eval_root: str,
    out_dir: str,
    ref_model: str = "transmorph_her",
    metrics: List[str] | None = None,
    min_subjects: int = 115,
) -> None:
    metrics = metrics or list(DEFAULT_METRICS)
    os.makedirs(out_dir, exist_ok=True)
    data = _discover(eval_root)
    if not data:
        raise FileNotFoundError(f"No per_case.csv found under {eval_root}")

    summary = _summary_rows(data, metrics)
    summary.to_csv(os.path.join(out_dir, "model_summary.csv"), index=False)

    # Keep only models with enough subjects for stable paired testing.
    data = {k: v for k, v in data.items() if len(v) >= min_subjects}
    if ref_model not in data:
        pd.DataFrame(
            [
                {
                    "ref_model": ref_model,
                    "baseline": "",
                    "metric": "",
                    "n": 0,
                    "p_raw": np.nan,
                    "q_bh_fdr": np.nan,
                    "significant_q<0.05": False,
                    "direction": "",
                    "note": f"Reference model `{ref_model}` has < {min_subjects} subjects in Eval_Results.",
                }
            ]
        ).to_csv(os.path.join(out_dir, "sig_matrix.csv"), index=False)
        return

    p_rows = _paired_pvals(data, ref_model=ref_model, metrics=metrics)
    pvals = [r.p_raw for r in p_rows if np.isfinite(r.p_raw)]
    qvals = _bh_fdr(pvals)
    qi = 0
    out_rows = []
    for r in p_rows:
        q = float("nan")
        if np.isfinite(r.p_raw):
            q = qvals[qi]
            qi += 1
        out_rows.append(
            {
                "ref_model": ref_model,
                "baseline": r.baseline,
                "metric": r.metric,
                "n": r.n,
                "p_raw": r.p_raw,
                "q_bh_fdr": q,
                "significant_q<0.05": bool(np.isfinite(q) and q < 0.05),
                "direction": "higher_better" if r.metric in HIGHER_IS_BETTER else "lower_better",
                "note": "",
            }
        )
    pd.DataFrame(out_rows).to_csv(os.path.join(out_dir, "sig_matrix.csv"), index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run paired Wilcoxon + BH-FDR on Eval_Results.")
    ap.add_argument("--eval_root", type=str, default=os.path.join("IXI", "Eval_Results"))
    ap.add_argument("--out_dir", type=str, default=os.path.join("IXI", "Results", "comprehensive"))
    ap.add_argument("--ref_model", type=str, default="transmorph_her")
    ap.add_argument("--min_subjects", type=int, default=115)
    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    eval_root = args.eval_root if os.path.isabs(args.eval_root) else os.path.join(repo_root, args.eval_root)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(repo_root, args.out_dir)
    run(
        eval_root=eval_root,
        out_dir=out_dir,
        ref_model=args.ref_model,
        min_subjects=args.min_subjects,
    )
    print(f"Wrote summary + significance tables to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
