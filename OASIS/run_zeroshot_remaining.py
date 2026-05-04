"""
Sequential runner for the remaining zero-shot OASIS steps.

Skips export/eval if the output already exists (idempotent).
Run AFTER transmorph_zs_oasis and transmorphbayes_zs_oasis are done:

    python OASIS/run_zeroshot_remaining.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

STEPS = [
    # (model_id, needs_export)
    ("transmorph_her_zs_oasis", True),
    ("pvt_zs_oasis", True),
]

# models whose eval results we want included in the final stats run
ALL_ZS_MODELS = [
    "transmorph_zs_oasis",
    "transmorphbayes_zs_oasis",
    "transmorph_her_zs_oasis",
    "pvt_zs_oasis",
]


def submission_dir(model_id: str) -> str:
    return os.path.join(REPO, "OASIS", "data", "Submit", "submission", model_id, "task_03")


def eval_dir(model_id: str) -> str:
    return os.path.join(REPO, "OASIS", "Eval_Results", model_id)


def export_done(model_id: str) -> bool:
    d = submission_dir(model_id)
    if not os.path.isdir(d):
        return False
    npz = [f for f in os.listdir(d) if f.endswith(".npz")]
    return len(npz) >= 19


def eval_done(model_id: str) -> bool:
    return os.path.isfile(os.path.join(eval_dir(model_id), "per_case.csv"))


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=REPO)
    elapsed = time.time() - t0
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed (exit {ret.returncode}): {' '.join(cmd)}")
    print(f"    done in {elapsed:.0f}s", flush=True)


def main() -> None:
    py = sys.executable

    for model_id, needs_export in STEPS:
        print(f"\n{'='*60}", flush=True)
        print(f"  MODEL: {model_id}", flush=True)
        print(f"{'='*60}", flush=True)

        if needs_export and not export_done(model_id):
            run([py, "OASIS/export_displacements.py", "--model-id", model_id])
        else:
            print(f"  [skip export] {submission_dir(model_id)} already has >=19 .npz files")

        if not eval_done(model_id):
            run([py, "OASIS/eval_oasis.py", "--models", model_id])
        else:
            print(f"  [skip eval]   {eval_dir(model_id)}/per_case.csv exists")

    print(f"\n{'='*60}", flush=True)
    print("  STATS: oasis_run_stats.py (all models incl. ZS)", flush=True)
    print(f"{'='*60}", flush=True)
    run([py, "OASIS/oasis_run_stats.py"])

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
