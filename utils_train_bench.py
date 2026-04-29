"""Training benchmark helpers: wall time and forward MACs (optional thop) for fair comparisons."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch


def profile_forward_macs_gparams_m(
    model: torch.nn.Module,
    img_size: Tuple[int, ...],
    device: torch.device,
    in_chans: int = 2,
    batch_size: int = 1,
) -> Tuple[Optional[float], float]:
    """
    One forward pass on a dummy input shaped [B, 2, D, H, W] (concatenated volumes).
    Returns (MACs in billions, params in millions). MACs is None if thop fails.
    """
    dummy = torch.zeros(
        batch_size, in_chans, *img_size, device=device, dtype=torch.float32
    )
    was_training = model.training
    model.eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    macs_g: Optional[float] = None
    try:
        from thop import profile  # type: ignore

        with torch.no_grad():
            macs, _p = profile(model, inputs=(dummy,), verbose=False)
        macs_g = float(macs) / 1e9
    except Exception as exc:
        print(f"[train_bench] thop profile skipped: {exc}")
    finally:
        if was_training:
            model.train()
    return macs_g, params_m


def log_benchmark_to_tensorboard(
    writer,
    macs_g: Optional[float],
    params_m: float,
    tag_mac: str = "Benchmark/MACs_forward_G",
    tag_params: str = "Benchmark/params_M",
    step: int = 0,
) -> None:
    writer.add_scalar(tag_params, params_m, step)
    if macs_g is not None:
        writer.add_scalar(tag_mac, macs_g, step)


def append_benchmark_json(log_dir: str, record: Dict[str, Any]) -> None:
    path = os.path.join(log_dir, "train_benchmark.json")
    rows: list = []
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                rows = [data]
        except json.JSONDecodeError:
            rows = []
    rows.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def perf_now() -> float:
    return time.perf_counter()
