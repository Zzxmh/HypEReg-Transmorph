"""
Parallel dispatch: one process per GPU, models processed in order (starmap chunks).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from typing import List, Optional, Sequence

from .config import (
    ModelSpec,
    ATLAS_PKL,
    DATA_ROOT_DEFAULT,
    MODEL_REGISTRY,
    get_ckpt_dir,
)
from .run_inference import run_one_model


def _one_job(
    spec: ModelSpec,
    gpu_id: int,
    out_root: str,
    test_dir: str,
    atlas_path: str,
    limit: Optional[int],
    resume: bool,
) -> None:
    try:
        run_one_model(
            spec, gpu_id, out_root, test_dir, atlas_path, limit, resume=resume
        )
    except Exception as e:
        mdir = get_ckpt_dir(spec)
        errp = os.path.join(out_root, spec.name, "_ERROR.log")
        os.makedirs(os.path.dirname(errp), exist_ok=True)
        with open(errp, "w") as ef:
            ef.write(f"{e}\n")
            ef.write(traceback.format_exc())
        print(f"[dispatch] {spec.name} failed: {e}")


def run_all_models(
    out_root: str,
    model_specs: Optional[Sequence[ModelSpec]] = None,
    gpus: Optional[Sequence[int]] = None,
    data_root: Optional[str] = None,
    test_subdir: str = "Test",
    limit: Optional[int] = None,
    resume: bool = False,
) -> None:
    data_root = data_root or DATA_ROOT_DEFAULT
    test_dir = os.path.join(data_root, test_subdir)
    if not test_dir.endswith("/"):
        test_dir = test_dir + "/"
    specs: List[ModelSpec] = list(model_specs) if model_specs is not None else list(MODEL_REGISTRY)
    g_list = [int(x) for x in (gpus or [0, 1])]
    n_proc = min(len(g_list), len(specs), max(1, len(g_list)))

    jobs = [
        (
            spec,
            g_list[i % len(g_list)],
            out_root,
            test_dir,
            ATLAS_PKL,
            limit,
            resume,
        )
        for i, spec in enumerate(specs)
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_proc) as pool:
        pool.starmap(_one_job, jobs)
