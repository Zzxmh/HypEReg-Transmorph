"""
Aggregate per-model summary.json into one model x metric table (CSV, MD, TeX).
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import ModelSpec, MODEL_REGISTRY


def _mstd(ps: Dict[str, Any], key: str) -> Tuple[float, float]:
    b = ps.get(key)
    if not isinstance(b, dict):
        return float("nan"), float("nan")
    return float(b.get("mean", float("nan"))), float(b.get("std", float("nan")))


def _mstd_voi(ps: Dict[str, Any], voi_key: str) -> Tuple[float, float]:
    b = ps.get(voi_key)
    if not isinstance(b, dict):
        return float("nan"), float("nan")
    inner = b.get("across_voi_per_subject", {})
    if not isinstance(inner, dict):
        return float("nan"), float("nan")
    return float(inner.get("mean", float("nan"))), float(inner.get("std", float("nan")))


def _fmt(m: float, t: float) -> str:
    if m != m:
        return "nan"
    if t == t and t > 0:
        return f"{m:.4f} ± {t:.4f}"
    return f"{m:.4f}"


def build_table(
    out_root: str,
    model_specs: Optional[Sequence[ModelSpec]] = None,
) -> Tuple[str, str, str]:
    specs = list(model_specs) if model_specs is not None else list(MODEL_REGISTRY)
    header = [
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
    data_rows: List[List[str]] = []
    for spec in specs:
        p = os.path.join(out_root, spec.name, "summary.json")
        if not os.path.isfile(p):
            data_rows.append([spec.name] + [""] * (len(header) - 1))
            continue
        with open(p) as f:
            s = json.load(f)
        ps: Dict[str, Any] = s.get("per_subject", {})
        inf: Dict[str, Any] = s.get("inference", {})

        def a(k: str) -> str:
            return _fmt(*_mstd(ps, k))

        def b(k: str) -> str:
            return _fmt(*_mstd_voi(ps, k))

        rt = inf.get("runtime_s_per_pair", {}) or {}
        pm = inf.get("peak_mem_GB", {}) or {}
        rtm = f"{rt.get('mean', float('nan')):.4f} ± {rt.get('std', float('nan')):.4f}" if isinstance(rt, dict) else ""
        pkm = f"{pm.get('max', float('nan')):.3f}" if isinstance(pm, dict) else ""

        row = [
            spec.name,
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
        data_rows.append(row)

    base = os.path.join(out_root, "table_model_x_metric")
    csv_p = base + ".csv"
    with open(csv_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data_rows)

    md_lines = [
        "# Model × metric (mean ± std where applicable)\n",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for r in data_rows:
        md_lines.append("| " + " | ".join(str(x) for x in r) + " |")
    md_p = base + ".md"
    with open(md_p, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    tex = [
        r"\begin{tabular}{" + "l" + "c" * (len(header) - 1) + "}",
        r"\hline",
        " & ".join(h.replace("_", r"\_") for h in header) + r"\\",
        r"\hline",
    ]
    for r in data_rows:
        tex.append(" & ".join(str(x) for x in r) + r"\\")
    tex += [r"\hline", r"\end{tabular}"]
    tex_p = base + ".tex"
    with open(tex_p, "w") as f:
        f.write("\n".join(tex) + "\n")

    return csv_p, md_p, tex_p
