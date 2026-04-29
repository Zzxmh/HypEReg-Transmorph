from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


def _bh_fdr(pvals: List[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def _paired_effect_signed(diff_aligned: np.ndarray) -> float:
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


def _common(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    inter = sorted(set(a["pkl"]).intersection(set(b["pkl"])))
    aa = a[a["pkl"].isin(inter)].copy().sort_values("pkl").reset_index(drop=True)
    bb = b[b["pkl"].isin(inter)].copy().sort_values("pkl").reset_index(drop=True)
    out = aa[["pkl"]].copy()
    out["her_dice"] = pd.to_numeric(aa["dice_mean"], errors="coerce")
    out["her_non_jec"] = pd.to_numeric(aa["non_jec"], errors="coerce")
    out["base_dice"] = pd.to_numeric(bb["dice_mean"], errors="coerce")
    out["base_non_jec"] = pd.to_numeric(bb["non_jec"], errors="coerce")
    return out


def run(eval_root: str, light_root: str, out_csv: str) -> None:
    her_path = os.path.join(eval_root, "transmorph_her", "per_case.csv")
    her = pd.read_csv(her_path)
    her = her[["pkl", "dice_mean", "non_jec"]].copy()

    baselines = {
        "transmorph_original_uploaded": os.path.join(light_root, "transmorph_original_light_per_case.csv"),
        "transmorphbayes_uploaded": os.path.join(light_root, "transmorphbayes_light_per_case.csv"),
    }

    rows: List[Dict[str, object]] = []
    pvals: List[float] = []
    for name, p in baselines.items():
        if not os.path.isfile(p):
            continue
        base = pd.read_csv(p)
        both = _common(her, base)
        for metric, hb, bb, higher_is_better in [
            ("dice_mean", "her_dice", "base_dice", True),
            ("non_jec", "her_non_jec", "base_non_jec", False),
        ]:
            a = both[hb].to_numpy(dtype=np.float64)
            b = both[bb].to_numpy(dtype=np.float64)
            m = np.isfinite(a) & np.isfinite(b)
            a = a[m]
            b = b[m]
            if a.size < 3:
                continue
            d_raw = a - b
            d_aligned = d_raw if higher_is_better else -d_raw
            p_raw = float(wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue)
            pvals.append(p_raw)
            rows.append(
                {
                    "metric": metric,
                    "baseline": name,
                    "n": int(a.size),
                    "her_mean": float(np.mean(a)),
                    "baseline_mean": float(np.mean(b)),
                    "median_paired_diff_raw": float(np.median(d_raw)),
                    "median_paired_diff_aligned": float(np.median(d_aligned)),
                    "effect_signed": float(_paired_effect_signed(d_aligned)),
                    "p_raw": p_raw,
                }
            )
    qvals = _bh_fdr(pvals)
    qi = 0
    for r in rows:
        r["q_bh_fdr"] = float(qvals[qi])
        r["significant_q<0.05"] = bool(float(qvals[qi]) < 0.05)
        qi += 1
    out = pd.DataFrame(rows).sort_values(["metric", "baseline"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote uploaded checkpoint inferential stats: {out_csv}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", default=os.path.join("IXI", "Eval_Results"))
    ap.add_argument("--light_root", default=os.path.join("IXI", "Results", "uploaded_weights_light"))
    ap.add_argument("--out_csv", default=os.path.join("IXI", "Results", "comprehensive", "sig_matrix_uploaded_ckpt.csv"))
    args = ap.parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    eval_root = args.eval_root if os.path.isabs(args.eval_root) else os.path.join(repo_root, args.eval_root)
    light_root = args.light_root if os.path.isabs(args.light_root) else os.path.join(repo_root, args.light_root)
    out_csv = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(repo_root, args.out_csv)
    run(eval_root=eval_root, light_root=light_root, out_csv=out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

