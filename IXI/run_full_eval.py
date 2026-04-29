# -*- coding: utf-8 -*-
"""
One-shot full IXI Test evaluation: infer + per_case metrics + aggregate.
Usage:
  python IXI/run_full_eval.py
  python IXI/run_full_eval.py --only transmorph_her
  python IXI/run_full_eval.py --skip transmorph_psc
"""
from __future__ import annotations

import argparse
import os
import sys
import time

IXI_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(IXI_DIR)
if IXI_DIR not in sys.path:
    sys.path.insert(0, IXI_DIR)


def main():
    ap = argparse.ArgumentParser(description="IXI full metric evaluation")
    ap.add_argument(
        "--config",
        default=os.path.join(IXI_DIR, "eval_configs.yaml"),
        help="Path to eval_configs.yaml",
    )
    ap.add_argument(
        "--only",
        default="",
        help="Comma-separated model ids to run (subset of eval_configs models)",
    )
    ap.add_argument(
        "--skip",
        default="",
        help="Comma-separated model ids to skip",
    )
    ap.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip aggregate_and_plot at the end",
    )
    ap.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit number of test cases (debug)",
    )
    args = ap.parse_args()

    try:
        import yaml
    except ImportError:
        print("Please: pip install PyYAML", file=sys.stderr)
        raise

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    from eval_any import run_one_model

    t0 = time.time()
    models = cfg.get("models", [])
    for entry in models:
        mid = entry.get("id")
        if only and mid not in only:
            continue
        if mid in skip:
            print("Skip", mid, flush=True)
            continue
        print("=" * 60, flush=True)
        print("Evaluating", mid, flush=True)
        run_one_model(entry, cfg, max_cases=args.max_cases)
    print("Total wall time (s):", time.time() - t0, flush=True)

    if not args.no_aggregate:
        from aggregate_and_plot import run as agg

        base = cfg.get("results_dir", "IXI/Eval_Results")
        if not os.path.isabs(base):
            base = os.path.join(REPO_ROOT, base)
        # augment legacy
        try:
            from augment_legacy_csv import run as aug
            rdir = cfg.get("legacy_results_dir", "IXI/Results")
            if not os.path.isabs(rdir):
                rdir = os.path.join(REPO_ROOT, rdir)
            aug(results_dir=rdir)
        except Exception as ex:  # pragma: no cover
            print("augment_legacy_csv:", ex, file=sys.stderr)
        bpath = base if os.path.isabs(base) else os.path.join(REPO_ROOT, base)
        agg(bpath)


if __name__ == "__main__":
    main()
