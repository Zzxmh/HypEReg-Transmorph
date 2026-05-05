# -*- coding: utf-8 -*-
"""
Merge IXI/Results/comprehensive/baseline table with the current HypEReg experiment row.
- Drops all baseline rows whose model name is HypEReg-related (detected via _is_her_related_model, case-insensitive).
- Appends one row from HER_dsc0743/summary.json (same formatting as build_table; folder name kept for filesystem compatibility).
Writes: IXI/Results/comprehensive/table_model_x_metric.{csv,md,tex}
Run: python -m IXI.analysis_comprehensive.merge_baseline_table
"""
from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any, Dict, List, Sequence

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_OUT = os.path.join(_REPO, "IXI", "Results", "comprehensive")
_DEFAULT_BASELINE = os.path.join(_DEFAULT_OUT, "baseline")
_HER_MODEL = "HER_dsc0743"


def _is_her_related_model(name: str) -> bool:
    return "her" in (name or "").lower()


def _row_from_summary_json(path: str, model_name: str) -> List[str]:
    from .build_table import _fmt, _mstd, _mstd_voi

    with open(path, encoding="utf-8") as f:
        s: Dict[str, Any] = json.load(f)
    ps: Dict[str, Any] = s.get("per_subject", {})
    inf: Dict[str, Any] = s.get("inference", {})

    def a(k: str) -> str:
        return _fmt(*_mstd(ps, k))

    def b(k: str) -> str:
        return _fmt(*_mstd_voi(ps, k))

    rt = inf.get("runtime_s_per_pair", {}) or {}
    pm = inf.get("peak_mem_GB", {}) or {}
    rtm = (
        f"{rt.get('mean', float('nan')):.4f} ± {rt.get('std', float('nan')):.4f}"
        if isinstance(rt, dict)
        else ""
    )
    pkm = f"{pm.get('max', float('nan')):.3f}" if isinstance(pm, dict) else ""

    return [
        model_name,
        b("dice_voi"),
        b("jaccard_voi"),
        b("vs_voi"),
        b("hd95_voi"),
        b("assd_voi"),
        b("nsd1mm_voi"),
        a("ncc"),
        a("lncc"),
        a("mi"),
        a("nmi"),
        a("ssim3d"),
        a("psnr"),
        a("non_jac_frac"),
        a("SDlogJ"),
        a("J_min"),
        a("J_p01"),
        a("J_p50"),
        a("J_p99"),
        a("J_max"),
        a("bending_energy"),
        a("mean_abs_div"),
        rtm,
        pkm,
    ]


def _header() -> List[str]:
    return [
        "model",
        "dice",
        "jaccard",
        "vs",
        "hd95_mm",
        "assd_mm",
        "nsd1mm",
        "ncc",
        "lncc",
        "mi",
        "nmi",
        "ssim3d",
        "psnr",
        "non_jac",
        "sdlogJ",
        "J_min",
        "J_p01",
        "J_p50",
        "J_p99",
        "J_max",
        "bending_energy",
        "mean_abs_div",
        "runtime_s",
        "peak_GB",
    ]


def merge(
    out_root: str = _DEFAULT_OUT,
    baseline_dir: str = _DEFAULT_BASELINE,
    her_model: str = _HER_MODEL,
) -> tuple[str, str, str]:
    base_csv = os.path.join(baseline_dir, "table_model_x_metric.csv")
    if not os.path.isfile(base_csv):
        raise FileNotFoundError(f"Missing baseline table: {base_csv}")
    her_sum = os.path.join(out_root, her_model, "summary.json")
    if not os.path.isfile(her_sum):
        raise FileNotFoundError(f"Missing HypEReg summary: {her_sum}")

    header = _header()
    kept: List[List[str]] = []
    with open(base_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Baseline CSV has no header")
        for d in reader:
            m = (d.get("model") or "").strip()
            if not m or _is_her_related_model(m):
                continue
            kept.append([str(d.get(h, "") or "") for h in header])

    her_row = _row_from_summary_json(her_sum, her_model)
    merged = kept + [her_row]

    base = os.path.join(out_root, "table_model_x_metric")
    csv_p = base + ".csv"
    with open(csv_p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(merged)

    md_lines = [
        "# Model × metric (baseline non-HypEReg + current HypEReg experiment)\n",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for r in merged:
        esc = [str(x).replace("|", "\\|") for x in r]
        md_lines.append("| " + " | ".join(esc) + " |")
    md_p = base + ".md"
    with open(md_p, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    tex = [
        r"\begin{tabular}{" + "l" + "c" * (len(header) - 1) + "}",
        r"\hline",
        " & ".join(h.replace("_", r"\_") for h in header) + r"\\",
        r"\hline",
    ]
    for row in merged:
        cells = []
        for x in row:
            s = str(x).replace("%", r"\%").replace("_", r"\_")
            cells.append(s)
        tex.append(" & ".join(cells) + r"\\")
    tex += [r"\hline", r"\end{tabular}"]
    tex_p = base + ".tex"
    with open(tex_p, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")

    return csv_p, md_p, tex_p


def main(argv: Sequence[str] | None = None) -> int:
    try:
        c, m, t = merge()
        print("Wrote merged table:")
        print(" ", c)
        print(" ", m)
        print(" ", t)
    except Exception as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
