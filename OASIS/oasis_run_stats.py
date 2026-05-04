"""
Paired Wilcoxon + Benjamini--Hochberg FDR on OASIS ``per_case.csv`` exports.

Reads ``OASIS/Eval_Results/<model_id>/per_case.csv`` (same schema as IXI) and writes:
  - ``OASIS/Eval_Results/_stats/oasis_summary.csv``
  - ``OASIS/Eval_Results/_stats/oasis_pairwise_wilcoxon.csv``

Run from repository root::

    python OASIS/oasis_run_stats.py
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    repo = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if repo not in sys.path:
        sys.path.insert(0, repo)

    ap = argparse.ArgumentParser(description="OASIS paired Wilcoxon + BH-FDR on Eval_Results.")
    ap.add_argument(
        "--eval-root",
        default=os.path.join(repo, "OASIS", "Eval_Results"),
        help="Directory containing one subfolder per model_id with per_case.csv",
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join(repo, "OASIS", "Eval_Results", "_stats"),
    )
    ap.add_argument("--ref-model", default="transmorph_her_oasis")
    ap.add_argument(
        "--min-subjects",
        type=int,
        default=12,
        help="Minimum paired cases required per model (OASIS val ~19--20 pairs).",
    )
    args = ap.parse_args()

    from IXI.analysis_comprehensive.stats import run

    run(
        eval_root=args.eval_root,
        out_dir=args.out_dir,
        ref_model=args.ref_model,
        min_subjects=args.min_subjects,
        summary_csv="oasis_summary.csv",
        pairwise_csv="oasis_pairwise_wilcoxon.csv",
    )
    print(f"Wrote oasis_summary.csv and oasis_pairwise_wilcoxon.csv under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
