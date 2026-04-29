from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_case(subject: str) -> Dict[str, np.ndarray]:
    ixi_data = os.path.join(_repo_root(), "IXI_data")
    tm_data_dir = os.path.join(_repo_root(), "IXI", "TransMorph", "data")
    if tm_data_dir not in sys.path:
        sys.path.insert(0, tm_data_dir)
    from data_utils import pkload  # noqa: WPS433

    atlas = os.path.join(ixi_data, "atlas.pkl")
    case = os.path.join(ixi_data, "Test", subject)
    x, _ = pkload(atlas)
    y, _ = pkload(case)
    return {"moving": x.astype(np.float32), "fixed": y.astype(np.float32)}


def run_one(adapter_name: str, moving: np.ndarray, fixed: np.ndarray, repeats: int) -> Dict[str, float]:
    ixi_dir = os.path.join(_repo_root(), "IXI")
    if ixi_dir not in sys.path:
        sys.path.insert(0, ixi_dir)
    ad = importlib.import_module(f"adapters.{adapter_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _cfg = ad.build_model(device=device)
    model.eval()
    x = torch.from_numpy(moving[None, None, ...]).to(device)
    y = torch.from_numpy(fixed[None, None, ...]).to(device)

    # warmup
    with torch.no_grad():
        for _ in range(5):
            _xw, _flow = ad.forward(model, x, y)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times: List[float] = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            _xw, _flow = ad.forward(model, x, y)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    peak_gb = 0.0
    if device == "cuda":
        peak_gb = float(torch.cuda.max_memory_allocated() / 1024.0**3)
    return {
        "mean_s": float(np.mean(times)),
        "median_s": float(np.median(times)),
        "std_s": float(np.std(times, ddof=0)),
        "peak_mem_gb": peak_gb,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Profile pure forward runtime and peak memory.")
    ap.add_argument("--subject", default="subject_1.pkl")
    ap.add_argument("--repeats", type=int, default=30)
    args = ap.parse_args()

    case = _load_case(args.subject)
    for name in ["transmorph_her", "transmorph_original"]:
        out = run_one(name, case["moving"], case["fixed"], repeats=args.repeats)
        print(
            f"{name}: mean={out['mean_s']:.4f}s median={out['median_s']:.4f}s "
            f"std={out['std_s']:.4f}s peak_mem={out['peak_mem_gb']:.3f}GB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

