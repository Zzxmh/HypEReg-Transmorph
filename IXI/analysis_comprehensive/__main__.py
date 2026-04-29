"""
CLI: python -m IXI.analysis_comprehensive
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

from .build_table import build_table
from .config import DATA_ROOT_DEFAULT, MODEL_REGISTRY, ModelSpec, repo_root
from .dispatch import run_all_models
from .plot_all import plot_cross_model


def _parse_only(s: str) -> Optional[Sequence[ModelSpec]]:
    if not s or not s.strip():
        return None
    want = {x.strip() for x in s.split(",") if x.strip()}
    out = [m for m in MODEL_REGISTRY if m.name in want]
    if not out:
        print("No model matched --only; use names like TransMorphBayes,HER_active", file=sys.stderr)
        sys.exit(1)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Comprehensive IXI registration metrics, parallel inference, tables, figures."
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output root (default: IXI/Results/comprehensive under repo root)",
    )
    ap.add_argument(
        "--skip_inference",
        action="store_true",
        help="Only rebuild table + figures from existing per-model outputs",
    )
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated model names (e.g. TransMorphBayes,HER_active)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of test subjects (for smoke test)",
    )
    ap.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="Comma-separated GPU ids for parallel model jobs",
    )
    ap.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Override IXI data root (default: repo/IXI_data)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing per-model CSVs; skip already-written subjects in dice.csv",
    )
    args = ap.parse_args(argv)
    out = args.out
    if not out:
        out = os.path.join(repo_root(), "IXI", "Results", "comprehensive")
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)

    specs = _parse_only(args.only)
    gpus: List[int] = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpus:
        gpus = [0]

    data_root = args.data_root or DATA_ROOT_DEFAULT

    if not args.skip_inference:
        run_all_models(
            out,
            model_specs=specs,
            gpus=gpus,
            data_root=data_root,
            limit=args.limit,
            resume=bool(args.resume),
        )

    build_table(out, model_specs=specs)
    plot_cross_model(out, specs=specs)
    print(f"Done. Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
