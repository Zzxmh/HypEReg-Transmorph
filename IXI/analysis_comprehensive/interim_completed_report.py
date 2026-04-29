"""
Build interim report artifacts from currently completed models.

Outputs to IXI/Results/comprehensive/interim/:
  - model_status.csv
  - table_interim.csv
  - table_interim.tex
  - figures/interim_accuracy.png
  - figures/interim_topology.png
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class Row:
    model: str
    source: str
    n: int
    dice_mean: Optional[float] = None
    dice_std: Optional[float] = None
    jaccard_mean: Optional[float] = None
    jaccard_std: Optional[float] = None
    hd95_mean: Optional[float] = None
    hd95_std: Optional[float] = None
    assd_mean: Optional[float] = None
    assd_std: Optional[float] = None
    non_jec_mean: Optional[float] = None
    non_jec_std: Optional[float] = None
    sdlogj_mean: Optional[float] = None
    sdlogj_std: Optional[float] = None
    j_p01_mean: Optional[float] = None
    j_p01_std: Optional[float] = None
    j_p99_mean: Optional[float] = None
    j_p99_std: Optional[float] = None
    inference_mean: Optional[float] = None
    inference_std: Optional[float] = None
    note: str = ""


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _mean_std(vals: List[float]):
    if not vals:
        return None, None
    a = np.asarray(vals, dtype=np.float64)
    return float(np.mean(a)), float(np.std(a, ddof=0))


def _from_eval_results(eval_root: str) -> Dict[str, Row]:
    out: Dict[str, Row] = {}
    if not os.path.isdir(eval_root):
        return out
    for m in sorted(os.listdir(eval_root)):
        p = os.path.join(eval_root, m, "per_case.csv")
        if not os.path.isfile(p):
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        n = len(rows)
        vals = lambda k: [float(r[k]) for r in rows if r.get(k) not in ("", None, "nan", "NaN")]
        r = Row(model=m, source=f"Eval_Results/{m}/per_case.csv", n=n)
        r.dice_mean, r.dice_std = _mean_std(vals("dice_mean"))
        r.jaccard_mean, r.jaccard_std = _mean_std(vals("jaccard_mean"))
        r.hd95_mean, r.hd95_std = _mean_std(vals("HD95_mean"))
        r.assd_mean, r.assd_std = _mean_std(vals("ASSD_mean"))
        r.non_jec_mean, r.non_jec_std = _mean_std(vals("non_jec"))
        r.sdlogj_mean, r.sdlogj_std = _mean_std(vals("SDlogJ"))
        r.j_p01_mean, r.j_p01_std = _mean_std(vals("J_p01"))
        r.j_p99_mean, r.j_p99_std = _mean_std(vals("J_p99"))
        r.inference_mean, r.inference_std = _mean_std(vals("inference_s"))
        out[m] = r
    return out


def _from_historical_table(table_csv: str) -> Dict[str, Row]:
    out: Dict[str, Row] = {}
    if not os.path.isfile(table_csv):
        return out
    df = pd.read_csv(table_csv)
    for _, rr in df.iterrows():
        model = str(rr["model"])
        if model not in ("HER_dsc0743", "TransMorphBayes"):
            continue
        row = Row(model=model, source="Results/comprehensive/table_model_x_metric.csv", n=115)
        def parse_pm(col: str):
            txt = str(rr[col])
            if "±" in txt:
                a, b = [t.strip() for t in txt.split("±", 1)]
                return _safe_float(a), _safe_float(b)
            v = _safe_float(txt)
            return v, None
        row.dice_mean, row.dice_std = parse_pm("dice")
        row.jaccard_mean, row.jaccard_std = parse_pm("jaccard")
        row.hd95_mean, row.hd95_std = parse_pm("hd95_mm")
        row.assd_mean, row.assd_std = parse_pm("assd_mm")
        row.non_jec_mean, row.non_jec_std = parse_pm("non_jac")
        row.sdlogj_mean, row.sdlogj_std = parse_pm("sdlogJ")
        row.j_p01_mean, row.j_p01_std = parse_pm("J_p01")
        row.j_p99_mean, row.j_p99_std = parse_pm("J_p99")
        row.inference_mean, row.inference_std = parse_pm("runtime_s")
        out[model] = row
    return out


def _from_legacy_dice(results_root: str) -> Dict[str, Row]:
    out: Dict[str, Row] = {}
    for fn, name in [("ants_IXI.csv", "syn_legacy"), ("affine.csv", "affine_legacy")]:
        p = os.path.join(results_root, fn)
        if not os.path.isfile(p):
            continue
        rows = list(csv.reader(open(p, encoding="utf-8")))[2:]
        dices, njs = [], []
        for r in rows:
            per = []
            for x in r[1:47]:
                try:
                    per.append(float(x))
                except Exception:
                    pass
            if per:
                dices.append(float(np.mean(per)))
            try:
                njs.append(float(r[47]))
            except Exception:
                pass
        rm, rs = _mean_std(dices)
        nm, ns = _mean_std(njs)
        out[name] = Row(
            model=name,
            source=f"Results/{fn}",
            n=len(dices),
            dice_mean=rm,
            dice_std=rs,
            non_jec_mean=nm,
            non_jec_std=ns,
            note="legacy Dice/non_jec only",
        )
    return out


def _fmt_pm(m: Optional[float], s: Optional[float], digits: int = 4) -> str:
    if m is None:
        return "N/A"
    if s is None:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def run() -> None:
    repo = _repo_root()
    eval_root = os.path.join(repo, "IXI", "Eval_Results")
    results_root = os.path.join(repo, "IXI", "Results")
    comp_root = os.path.join(results_root, "comprehensive")
    out_root = os.path.join(comp_root, "interim")
    fig_root = os.path.join(out_root, "figures")
    os.makedirs(fig_root, exist_ok=True)

    eval_rows = _from_eval_results(eval_root)
    hist_rows = _from_historical_table(os.path.join(comp_root, "table_model_x_metric.csv"))
    legacy_rows = _from_legacy_dice(results_root)

    # status
    status_rows = []
    for m, r in sorted(eval_rows.items()):
        status_rows.append({"model": m, "source": r.source, "cases": r.n, "completed_115": int(r.n >= 115)})
    pd.DataFrame(status_rows).to_csv(os.path.join(out_root, "model_status.csv"), index=False)

    # include fully complete eval models + historical her/bayes + legacy references
    rows: List[Row] = []
    for m, r in sorted(eval_rows.items()):
        if r.n >= 115:
            rows.append(r)
    for k in ("TransMorphBayes", "HER_dsc0743"):
        if k in hist_rows:
            rows.append(hist_rows[k])
    for k in ("syn_legacy", "affine_legacy"):
        if k in legacy_rows:
            rows.append(legacy_rows[k])

    tbl = []
    for r in rows:
        tbl.append(
            {
                "model": r.model,
                "source": r.source,
                "n": r.n,
                "dice": _fmt_pm(r.dice_mean, r.dice_std, 4),
                "jaccard": _fmt_pm(r.jaccard_mean, r.jaccard_std, 4),
                "hd95_mm": _fmt_pm(r.hd95_mean, r.hd95_std, 4),
                "assd_mm": _fmt_pm(r.assd_mean, r.assd_std, 4),
                "non_jec": _fmt_pm(r.non_jec_mean, r.non_jec_std, 6),
                "sdlogJ": _fmt_pm(r.sdlogj_mean, r.sdlogj_std, 4),
                "J_p01": _fmt_pm(r.j_p01_mean, r.j_p01_std, 4),
                "J_p99": _fmt_pm(r.j_p99_mean, r.j_p99_std, 4),
                "runtime_s": _fmt_pm(r.inference_mean, r.inference_std, 4),
                "note": r.note,
            }
        )
    tdf = pd.DataFrame(tbl)
    tdf.to_csv(os.path.join(out_root, "table_interim.csv"), index=False)

    # tex table
    tex_path = os.path.join(out_root, "table_interim.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\hline\n")
        f.write("model & dice & hd95\\_mm & assd\\_mm & non\\_jec & sdlogJ & runtime\\_s\\\\\n")
        f.write("\\hline\n")
        for _, rr in tdf.iterrows():
            f.write(
                f"{rr['model']} & {rr['dice']} & {rr['hd95_mm']} & {rr['assd_mm']} & "
                f"{rr['non_jec']} & {rr['sdlogJ']} & {rr['runtime_s']}\\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    # plots (numeric subset)
    numeric_rows = [r for r in rows if r.dice_mean is not None and r.hd95_mean is not None and r.assd_mean is not None]
    if numeric_rows:
        names = [r.model for r in numeric_rows]
        dice = [r.dice_mean for r in numeric_rows]
        hd95 = [r.hd95_mean for r in numeric_rows]
        assd = [r.assd_mean for r in numeric_rows]
        x = np.arange(len(names))

        fig, ax = plt.subplots(1, 3, figsize=(14, 4), dpi=150)
        ax[0].bar(x, dice)
        ax[0].set_title("Dice (higher better)")
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(names, rotation=45, ha="right")
        ax[1].bar(x, hd95)
        ax[1].set_title("HD95 mm (lower better)")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(names, rotation=45, ha="right")
        ax[2].bar(x, assd)
        ax[2].set_title("ASSD mm (lower better)")
        ax[2].set_xticks(x)
        ax[2].set_xticklabels(names, rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(os.path.join(fig_root, "interim_accuracy.png"), bbox_inches="tight")
        plt.close(fig)

    topology_rows = [r for r in rows if r.non_jec_mean is not None and r.sdlogj_mean is not None]
    if topology_rows:
        names = [r.model for r in topology_rows]
        nonj = [r.non_jec_mean for r in topology_rows]
        sdlog = [r.sdlogj_mean for r in topology_rows]
        x = np.arange(len(names))
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
        ax[0].bar(x, nonj)
        ax[0].set_title("non_jec (lower better)")
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(names, rotation=45, ha="right")
        ax[1].bar(x, sdlog)
        ax[1].set_title("SDlogJ (lower better)")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(names, rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(os.path.join(fig_root, "interim_topology.png"), bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote interim artifacts under: {out_root}")


if __name__ == "__main__":
    run()
