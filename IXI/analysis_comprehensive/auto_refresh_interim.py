from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from datetime import datetime


TARGET_MODELS = ["voxelmorph_1", "cyclemorph", "midir", "cotr", "nnformer", "pvt", "syn", "affine"]


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def case_count(eval_root: str, model: str) -> int:
    p = os.path.join(eval_root, model, "per_case.csv")
    if not os.path.isfile(p):
        return 0
    with open(p, encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def run_py(script: str, *args: str) -> int:
    cmd = [sys.executable, script, *args]
    return subprocess.call(cmd, cwd=repo_root())


def main() -> int:
    root = repo_root()
    eval_root = os.path.join(root, "IXI", "Eval_Results")
    out_dir = os.path.join(root, "IXI", "Results", "comprehensive")
    log_path = os.path.join(out_dir, "interim", "auto_refresh.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    interval_s = 600
    while True:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        counts = {m: case_count(eval_root, m) for m in TARGET_MODELS}

        run_py(os.path.join(root, "IXI", "analysis_comprehensive", "interim_completed_report.py"))
        run_py(
            os.path.join(root, "IXI", "analysis_comprehensive", "stats.py"),
            "--eval_root",
            "IXI/Eval_Results",
            "--out_dir",
            "IXI/Results/comprehensive",
            "--ref_model",
            "transmorph_her",
            "--min_subjects",
            "115",
        )

        line = f"{ts} " + ", ".join([f"{k}={counts[k]}" for k in TARGET_MODELS])
        with open(log_path, "a", encoding="utf-8", newline="") as f:
            f.write(line + "\n")
        print(line)

        if all(v >= 115 for v in counts.values()):
            print("All target models reached >=115 cases. Exiting auto refresh.")
            return 0
        time.sleep(interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
