# -*- coding: utf-8 -*-
"""
Smoke test: 1 Test case TransMorph+HER — dice_* and non_jec must match IXI/Results/TransMorph_HER_IXI.csv p_0 row.
Run from repo root: python IXI/smoke_test_full_eval.py
Requires GPU + IXI_data Test + checkpoint.
"""
from __future__ import annotations

import csv
import os
import sys

IXI_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(IXI_DIR)
if IXI_DIR not in sys.path:
    sys.path.insert(0, IXI_DIR)


def main():
    try:
        import yaml
    except ImportError:
        print("pip install PyYAML", file=sys.stderr)
        return 1
    with open(
        os.path.join(IXI_DIR, "eval_configs.yaml"), "r", encoding="utf-8"
    ) as f:
        cfg = yaml.safe_load(f)
    entry = next(
        m for m in cfg["models"] if m.get("id") == "transmorph_her"
    )
    from eval_any import run_one_model

    run_one_model(entry, cfg, max_cases=1)
    res_csv = os.path.join(
        REPO_ROOT, "IXI", "Eval_Results", "transmorph_her", "per_case.csv"
    )
    her_ref = os.path.join(IXI_DIR, "Results", "TransMorph_HER_IXI.csv")
    with open(res_csv, "r", encoding="utf-8") as f:
        r = list(csv.DictReader(f))
    with open(her_ref, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    # First data line p_0
    p0 = [x.strip() for x in lines[2].split(",")]

    d_new = {f"dice_{i}": float(r[0][f"dice_{i}"]) for i in range(46)}
    nj_new = float(r[0]["non_jec"])
    d_ref = {f"dice_{i}": float(p0[1 + i]) for i in range(46)}
    nj_ref = float(p0[-1])

    tol = 1e-5
    max_d = max(abs(d_new[f"dice_{i}"] - d_ref[f"dice_{i}"]) for i in range(46))
    if max_d > tol or abs(nj_new - nj_ref) > tol:
        print("MISMATCH: max dice diff =", max_d, "non_jec diff =", abs(nj_new - nj_ref))
        return 1
    print("OK: per_case matches TransMorph_HER_IXI.csv p_0 (tol={})".format(tol))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
