# -*- coding: utf-8 -*-
"""
Summarize per_case.csv across models, Wilcoxon + Bonferroni, figures, summary.md/csv.
"""
from __future__ import annotations

import json
import os
import sys
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np

IXI_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(IXI_DIR)

try:
    import matplotlib
    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

from scipy import stats
from scipy.stats import wilcoxon

# Merged 17 groups (analysis_trans outstruct)
_OUTSTRUCT = [
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


def _read_results_header_46() -> List[str]:
    ref = os.path.join(IXI_DIR, "Results", "TransMorph_HER_IXI.csv")
    if not os.path.isfile(ref):
        return [f"col_{i}" for i in range(46)]
    with open(ref, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    if len(lines) < 2:
        return [f"col_{i}" for i in range(46)]
    parts = [p.strip() for p in lines[1].split(",")]
    if len(parts) < 48:
        return [f"col_{i}" for i in range(46)]
    return parts[1:47]


def outstruct_to_dice_col_indices() -> List[List[int]]:
    """For each of 17 merged names, average dice columns (analysis_trans: stct in item)."""
    names = _read_results_header_46()
    return [[j for j, item in enumerate(names) if st in item] for st in _OUTSTRUCT]


def _t_ci(x: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float, float, float]:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        m = float(np.mean(x)) if n else float("nan")
        return m, float("nan"), m, m, m
    m, se = float(np.mean(x)), float(stats.sem(x, nan_policy="omit"))
    h = stats.t.ppf(1 - alpha / 2, n - 1) * se
    return m, float(np.std(x, ddof=1)), m - h, m + h, se


def load_per_case_dir(path: str) -> "pd.DataFrame":
    p = os.path.join(path, "per_case.csv")
    if not os.path.isfile(p):
        p = path  # file path
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required: pip install pandas")
    return pd.read_csv(p)


def discover_models(
    base: str, legacy_sub: str = "legacy_from_results"
) -> List[Tuple[str, str, str]]:
    """
    (model_id, path to dir or per_case, label) — prefer Eval_Results/* subdirs; then legacy.
    """
    el = []
    b = base if os.path.isabs(base) else os.path.join(IXI_DIR, base)
    for d in sorted(glob(os.path.join(b, "*"))):
        if not os.path.isdir(d):
            continue
        m = os.path.basename(d)
        if m in (legacy_sub, "figures", "__pycache__"):
            continue
        pc = os.path.join(d, "per_case.csv")
        if os.path.isfile(pc):
            el.append((m, d, m))
    leg = os.path.join(b, legacy_sub)
    if os.path.isdir(leg):
        for f in glob(os.path.join(leg, "*_per_case.csv")):
            m = os.path.splitext(os.path.basename(f).replace("_per_case", ""))[0]
            if m.endswith("_per_case"):
                m = f.replace(".csv", "").replace("_per_case", "")
            el.append(
                (
                    "legacy__" + os.path.splitext(os.path.basename(f))[0].replace(
                        "_per_case", ""
                    ),
                    f,
                    os.path.splitext(os.path.basename(f).replace("_per_case", ""))[0],
                )
            )
    return el


def _mean_dice_merged(
    df: "pd.DataFrame", idxs_per_struct: List[List[int]]
) -> np.ndarray:
    n_subj = len(df)
    out = np.zeros((len(_OUTSTRUCT), n_subj))
    for si, idxs in enumerate(idxs_per_struct):
        if not idxs:
            out[si, :] = np.nan
            continue
        cols = [f"dice_{j}" for j in idxs]
        sub = df[cols].values.astype(np.float64)
        with np.errstate(invalid="ignore"):
            out[si] = np.nanmean(sub, axis=1)
    return out


def run(
    eval_base: str = "Eval_Results",
    ref_model: str = "transmorph_original",
) -> None:
    if plt is None or pd is None:  # pragma: no cover
        raise RuntimeError("Need pandas + matplotlib: pip install -r IXI/requirements-ixi-eval.txt")
    b = (
        os.path.normpath(eval_base)
        if os.path.isabs(eval_base)
        else os.path.join(IXI_DIR, eval_base)
    )
    os.makedirs(b, exist_ok=True)
    figd = os.path.join(b, "figures")
    os.makedirs(figd, exist_ok=True)

    idxs = outstruct_to_dice_col_indices()
    models = discover_models(b)
    if not models:
        print("No per_case.csv found under", b, file=sys.stderr)
        return

    # Load dataframes
    data: Dict[str, "pd.DataFrame"] = {}
    labels: Dict[str, str] = {}
    for mid, path, lab in models:
        data[mid] = load_per_case_dir(path)
        labels[mid] = lab

    # Summary metrics: dice_mean, all scalars
    keys = [
        "dice_mean",
        "HD95_mean",
        "ASSD_mean",
        "SSIM",
        "NMI",
        "non_jec",
        "SDlogJ",
        "J_p01",
        "J_p99",
        "inference_s",
    ]
    rows = []
    for mid, df in data.items():
        r = {"model": mid, "label": labels.get(mid, mid)}
        for k in keys:
            if k in df.columns:
                v = df[k].values.astype(np.float64)
                m, s, lo, hi, _ = _t_ci(v)
                r[f"{k}_mean"] = m
                r[f"{k}_std"] = s
                r[f"{k}_ci95_l"] = lo
                r[f"{k}_ci95_h"] = hi
            else:
                r[k] = np.nan
        rows.append(r)
    s_df = pd.DataFrame(rows)
    s_df.to_csv(os.path.join(b, "summary.csv"), index=False)

    # Markdown table
    lines = ["# Summary (mean, 95% CI where applicable)\n", "| model | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for mid, df in data.items():
        r = f"| {labels.get(mid, mid)} |"
        for k in keys:
            if k not in df.columns:
                r += " — |"
                continue
            v = df[k].values.astype(np.float64)
            m, _s, lo, hi, _ = _t_ci(v)
            r += f" {m:.4f} [{lo:.4f}, {hi:.4f}] |" if k != "dice_mean" or np.isfinite(m) else " — |"
        lines.append(r)
    with open(os.path.join(b, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Wilcoxon: dice_mean vs ref
    ref = next(
        (m[0] for m in models if m[0] == ref_model or ref_model in m[0]),
        models[0][0] if models else None,
    )
    ref_ser = data[ref]["dice_mean"].values.astype(np.float64) if ref in data else None
    if ref_ser is not None:
        wlines = [f"Wilcoxon vs reference `{ref_model}` (Bonferroni, k={len(data)-1}):"]
        k = max(len(data) - 1, 1)
        for mid, df in data.items():
            if mid == ref:
                continue
            a = ref_ser[: min(len(ref_ser), len(df)) ]
            c = df["dice_mean"].values.astype(np.float64)[: len(a)]
            try:
                w = wilcoxon(a, c, alternative="two-sided", mode="auto")
                p = min(1.0, w.pvalue * k)
            except Exception as ex:  # pragma: no cover
                p, ex = float("nan"), ex
            wlines.append(
                f"  {mid}: p_adj ≈ {p}"
                if np.isfinite(p)
                else f"  {mid}: p_adj = nan"
            )
        with open(os.path.join(b, "wilcoxon_dice.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(wlines) + "\n")

    # Box merged Dice (17 groups × models, analysis_trans-style offsets)
    n_struct = len(_OUTSTRUCT)
    n_mod = max(len(models), 1)
    spacing = 8.0
    sep = 0.6
    plt.figure(figsize=(16, 8), dpi=120)
    ax = plt.gca()
    for j, (mid, _, _) in enumerate(models):
        merged = _mean_dice_merged(data[mid], idxs)  # (S, n_subj)
        for i in range(n_struct):
            pos = i * spacing + (j - (n_mod - 1) / 2) * sep
            ax.boxplot(merged[i], positions=[pos], widths=0.45)
    tick_pos = [i * spacing for i in range(n_struct)]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(_OUTSTRUCT, rotation=90)
    ax.grid(True, axis="y", ls="--", alpha=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_title("Merged Dice (17 structure groups) — side-by-side by model")
    plt.tight_layout()
    pfp = os.path.join(figd, "box_dice_merged.png")
    plt.savefig(pfp, bbox_inches="tight")
    plt.close()
    print("Saved", pfp)

    # Volume similarity (VS) — same merged groups, vs_* columns
    if any("vs_0" in data[m].columns for m in data):
        n_struct = len(_OUTSTRUCT)
        n_mod = max(len(models), 1)
        spacing, sep = 8.0, 0.6
        plt.figure(figsize=(16, 8), dpi=120)
        ax = plt.gca()
        for j, (mid, _, _) in enumerate(models):
            df = data[mid]
            if "vs_0" not in df.columns:
                continue
            mvs = np.zeros((n_struct, len(df)))
            for si, cidxs in enumerate(idxs):
                if not cidxs:
                    mvs[si, :] = np.nan
                    continue
                cols = [f"vs_{k}" for k in cidxs]
                sub = df[cols].values.astype(np.float64)
                with np.errstate(invalid="ignore"):
                    mvs[si] = np.nanmean(sub, axis=1)
            for i in range(n_struct):
                pos = i * spacing + (j - (n_mod - 1) / 2) * sep
                ax.boxplot(mvs[i], positions=[pos], widths=0.45)
        tick_pos = [i * spacing for i in range(n_struct)]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(_OUTSTRUCT, rotation=90)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title("Merged Volume Similarity (17 groups)")
        ax.grid(True, axis="y", ls="--", alpha=0.5)
        plt.tight_layout()
        pvs = os.path.join(figd, "box_vs_by_voi.png")
        plt.savefig(pvs, bbox_inches="tight")
        plt.close()
        print("Saved", pvs)

    # Jacobian histogram (sum first 5 per-case hists in first available flow_stats.json)
    fs = None
    for c in glob(os.path.join(b, "**/flow_stats.json"), recursive=True):
        fs = c
        break
    if fs and os.path.isfile(fs):
        with open(fs, "r", encoding="utf-8") as jf:
            fsj = json.load(jf)
        cas = fsj.get("cases", [])
        hsum, edges, mid = None, None, os.path.dirname(fs)
        for c in cas[:5]:
            h = np.array(c.get("hist_counts", []), dtype=np.float64)
            edges = c.get("hist_edges")
            if hsum is None and len(h) > 0:
                hsum = h
            elif h is not None and hsum is not None and len(h) == len(hsum):
                hsum = hsum + h
        if hsum is not None and len(hsum) > 0 and edges is not None:
            plt.figure(figsize=(8, 4), dpi=120)
            cbin = 0.5 * (np.array(edges[1:]) + np.array(edges[:-1]))
            plt.plot(cbin, hsum / (hsum.sum() + 1e-9), label=mid)
            plt.xlabel("J")
            plt.ylabel("p(J)")
            plt.title("Jacobian histogram (sum first cases)")
            plt.tight_layout()
            plt.savefig(
                os.path.join(figd, "hist_jacobian.png"), bbox_inches="tight"
            )
            plt.close()

    # Scatter log(non_jec) vs dice_mean
    plt.figure(figsize=(7, 5), dpi=120)
    for mid, path, _ in models:
        df = data[mid]
        if "non_jec" in df.columns and "dice_mean" in df.columns:
            y = -np.log10(np.asarray(df["non_jec"], dtype=np.float64) + 1e-12)
            x = np.asarray(df["dice_mean"], dtype=np.float64)
            plt.scatter(x, y, s=4, label=mid[:20], alpha=0.5)
    plt.xlabel("dice_mean")
    plt.ylabel("-log10(non_jec)")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend(fontsize=6, ncols=2, loc="best")
    plt.tight_layout()
    plt.savefig(
        os.path.join(figd, "scatter_dice_vs_logNonJec.png"), bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Eval_Results")
    ap.add_argument("--ref", default="transmorph_original")
    a = ap.parse_args()
    run(a.base, a.ref)
