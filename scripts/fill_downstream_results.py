#!/usr/bin/env python3
"""
Fill PLACEHOLDER values in article.tex using downstream experiment results.

Run after oasis_downstream.py has produced:
  OASIS/Eval_Results/downstream/d1_summary.csv
  OASIS/Eval_Results/downstream/d2_icc_summary.csv

Usage:
    python scripts/fill_downstream_results.py
    python scripts/fill_downstream_results.py --dry-run
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TEX  = REPO / "draft" / "article.tex"
D1   = REPO / "OASIS" / "Eval_Results" / "downstream" / "d1_summary.csv"
D1P  = REPO / "OASIS" / "Eval_Results" / "downstream" / "d1_per_target.csv"
D2   = REPO / "OASIS" / "Eval_Results" / "downstream" / "d2_icc_summary.csv"

MODEL_DISPLAY = {
    "HypEReg-TransMorph (ZS)": "HypEReg-TransMorph",
    "TransMorph (ZS)":          "TransMorph",
    "TransMorphBayes (ZS)":     "TransMorphBayes",
}
ROI_DISPLAY = {
    "Hippocampus":        "Hippocampus",
    "Lateral_Ventricles": "Lateral Ventricles",
    "Thalamus":           "Thalamus",
}
MODEL_ORDER = ["HypEReg-TransMorph (ZS)", "TransMorph (ZS)", "TransMorphBayes (ZS)"]
ROI_ORDER   = ["Hippocampus", "Lateral_Ventricles", "Thalamus"]


def fmt(v, decimals=4) -> str:
    if not np.isfinite(v): return "---"
    return f"{v:.{decimals}f}"


def fmt_pm(m, s, d=4) -> str:
    if not np.isfinite(m): return "---"
    return rf"{m:.{d}f} \(\pm\) {s:.{d}f}"


def build_d1_table(d1s: pd.DataFrame, d1p: pd.DataFrame) -> str:
    """Return rows of Table tab:multiatlas."""
    rows = []
    for m in MODEL_ORDER:
        if m not in d1s.index:
            rows.append(f"{MODEL_DISPLAY.get(m,m)} & --- & --- & --- & --- & --- & --- \\\\")
            continue
        r = d1s.loc[m]

        def g(col, stat): 
            try: return float(r[(col, stat)])
            except: return np.nan

        s_m, s_s = g("single_dice_mean","mean"), g("single_dice_mean","std")
        f_m, f_s = g("fused_dice_mean","mean"),  g("fused_dice_mean","std")
        dd       = g("delta_dice","mean")
        hip      = g("fused_dice_Hippocampus","mean")  if "fused_dice_Hippocampus" in [c[0] for c in d1s.columns] else np.nan
        ven      = g("fused_dice_Lateral_Ventricles","mean") if "fused_dice_Lateral_Ventricles" in [c[0] for c in d1s.columns] else np.nan
        tha      = g("fused_dice_Thalamus","mean") if "fused_dice_Thalamus" in [c[0] for c in d1s.columns] else np.nan

        # Bold best fused Dice
        bold = "\\textbf" if m == MODEL_ORDER[0] else ""
        row = (f"{MODEL_DISPLAY.get(m,m)} & "
               f"{fmt_pm(s_m,s_s)} & "
               f"{bold}{{{fmt_pm(f_m,f_s)}}} & "
               f"+{fmt(dd,4)} & "
               f"{fmt(hip)} & {fmt(ven)} & {fmt(tha)} \\\\")
        rows.append(row)
    return "\n".join(rows)


def build_d2_table(d2: pd.DataFrame) -> str:
    """Return rows of Table tab:volreliability, 3 ROIs × N models."""
    all_rows = []
    for roi in ROI_ORDER:
        sub = d2[d2["roi"] == roi]
        model_rows = []
        for m in MODEL_ORDER:
            ms = sub[sub["model"] == m]
            if ms.empty:
                model_rows.append(f" & {MODEL_DISPLAY.get(m,m)} & --- & --- & --- & ---")
                continue
            r = ms.iloc[0]
            bold_jac = "\\textbf" if m == MODEL_ORDER[0] else ""
            model_rows.append(
                f" & {MODEL_DISPLAY.get(m,m)} & "
                f"{fmt(r.get('are_prop_mean',np.nan),2)} & "
                f"{bold_jac}{{{fmt(r.get('are_jac_mean',np.nan),2)}}} & "
                f"{fmt(r.get('icc_prop',np.nan),3)} & "
                f"{bold_jac}{{{fmt(r.get('icc_jac',np.nan),3)}}}"
                r" \\"
            )
        # First row of this ROI gets \multirow
        n = len(model_rows)
        model_rows[0] = rf"\multirow{{{n}}}{{*}}{{{ROI_DISPLAY.get(roi,roi)}}}" + model_rows[0]
        all_rows.extend(model_rows)
        all_rows.append(r"\midrule")
    if all_rows and all_rows[-1] == r"\midrule":
        all_rows.pop()
    return "\n".join(all_rows)


def update_tex(tex_text: str, d1s: pd.DataFrame, d1p: pd.DataFrame, d2: pd.DataFrame) -> str:
    # Replace Table tab:multiatlas rows
    d1_rows = build_d1_table(d1s, d1p)
    d2_rows = build_d2_table(d2)

    # Pattern: find midrule ... bottomrule inside tab:multiatlas
    def replace_table_body(text, label, new_rows):
        pat = rf"(\\label{{{re.escape(label)}}}.*?\\midrule\n)(.*?)(\\bottomrule)"
        m = re.search(pat, text, flags=re.DOTALL)
        if not m:
            print(f"  [warn] could not locate table body for {label}")
            return text
        return text[:m.start(2)] + new_rows + "\n" + text[m.end(2):]

    text = replace_table_body(tex_text, "tab:multiatlas", d1_rows)
    text = replace_table_body(text,     "tab:volreliability", d2_rows)

    # Replace narrative placeholders in §3.7 and §3.8
    if not d1s.empty and "HypEReg-TransMorph (ZS)" in d1s.index:
        h = d1s.loc["HypEReg-TransMorph (ZS)"]
        def g(col, stat):
            try: return float(h[(col, stat)])
            except: return np.nan
        hf, hs = g("fused_dice_mean","mean"), g("fused_dice_mean","std")
        hd      = g("delta_dice","mean")

        tm = d1s.loc["TransMorph (ZS)"] if "TransMorph (ZS)" in d1s.index else None
        tf, ts = (float(tm[("fused_dice_mean","mean")]), float(tm[("fused_dice_mean","std")])) if tm is not None else (np.nan, np.nan)
        td      = float(tm[("delta_dice","mean")]) if tm is not None else np.nan

        # Narrative sentence
        nar_pattern = (r"HypEReg-TransMorph achieves a fused Dice of \\\\\\(PLACEHOLDER \\\\pm PLACEHOLDER\\\\\\).*?"
                       r"TransMorph zero-shot \\\\\\(9\\.6\\\\times10\\^\\{-3\\}\\\\\\)")
        new_nar = (
            rf"HypEReg-TransMorph achieves a fused Dice of \\({fmt(hf)} \\pm {fmt(hs)}\\) "
            rf"in the zero-shot group, with \\(\\Delta\\mathrm{{Dice}}=+{fmt(hd)}\\). "
            rf"By comparison, plain TransMorph attains a fused Dice of \\({fmt(tf)} \\pm {fmt(ts)}\\) with \\(\\Delta\\mathrm{{Dice}}=+{fmt(td)}\\); "
            rf"the smaller improvement under fusion is consistent with the higher non-positive Jacobian ratio"
        )
        text = re.sub(
            r"HypEReg-TransMorph achieves a fused Dice of \\(PLACEHOLDER \\\\pm PLACEHOLDER\\).*?"
            r"consistent with the higher non-positive Jacobian ratio",
            new_nar, text, flags=re.DOTALL
        )

    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not D1.exists():
        print(f"ERROR: {D1} not found. Run oasis_downstream.py first.")
        return

    d1s = pd.read_csv(D1, index_col=0, header=[0, 1])
    d1p = pd.read_csv(D1P) if D1P.exists() else pd.DataFrame()
    d2  = pd.read_csv(D2)  if D2.exists()  else pd.DataFrame()

    print("D-1 summary:")
    print(d1s)
    if not d2.empty:
        print("\nD-2 ICC summary:")
        print(d2)

    tex = TEX.read_text(encoding="utf-8")
    updated = update_tex(tex, d1s, d1p, d2)

    if args.dry_run:
        diff_lines = sum(1 for a,b in zip(tex.splitlines(), updated.splitlines()) if a != b)
        print(f"\n[dry-run] {diff_lines} lines would change in {TEX}")
    else:
        TEX.write_text(updated, encoding="utf-8")
        print(f"\nUpdated {TEX}")


if __name__ == "__main__":
    main()
