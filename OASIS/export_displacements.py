"""
Export half-resolution displacement fields for OASIS Learn2Reg-style pairs.

Writes ``disp_{fixed:04d}_{moving:04d}.npz`` under
``OASIS/data/Submit/submission/{model_id}/task_03/`` for consumption by ``eval_oasis.py``.

Usage (from repository root)::

    python OASIS/export_displacements.py --model-id transmorph_her_oasis
    python OASIS/eval_oasis.py --models transmorph_her_oasis
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

import numpy as np
import torch
import yaml
from scipy.ndimage import zoom


def _repo_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_adapter(model_id: str):
    if model_id not in {
        "transmorph_her_oasis",
        "transmorph_unsup_oasis",
        "transmorph_dsc857_oasis",
        "voxelmorph_1_oasis",
        "midir_oasis",
        "cyclemorph_oasis",
        "syn_oasis",
        "affine_oasis",
        # IXI-trained zero-shot transfer
        "transmorph_zs_oasis",
        "transmorphbayes_zs_oasis",
        "transmorph_her_zs_oasis",
        "cotr_zs_oasis",
        "nnformer_zs_oasis",
        "pvt_zs_oasis",
    }:
        raise ValueError(f"Unknown model_id: {model_id}")
    return importlib.import_module(f"OASIS.adapters.{model_id}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export OASIS displacement fields for one model_id.")
    ap.add_argument("--model-id", required=True)
    ap.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "eval_configs.yaml"),
    )
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    args = ap.parse_args()

    repo = _repo_root()
    if repo not in sys.path:
        sys.path.insert(0, repo)

    cfg = _load_cfg(args.config if os.path.isabs(args.config) else os.path.join(repo, args.config))
    oasis_root = cfg.get("oasis_root", "OASIS/data")
    if not os.path.isabs(oasis_root):
        oasis_root = os.path.join(repo, oasis_root)
    test_dir = os.path.join(oasis_root, "Test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"OASIS Test dir missing: {test_dir}")

    out_root = os.path.join(repo, "OASIS", "data", "Submit", "submission", args.model_id, "task_03")
    os.makedirs(out_root, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mod = _load_adapter(args.model_id)
    model, _cfg = mod.build_model(device)
    model.eval()

    # OASIS/TransMorph/utils.pkload (moving, fixed, ...)
    tm_dir = os.path.join(repo, "OASIS", "TransMorph")
    if tm_dir not in sys.path:
        sys.path.insert(0, tm_dir)
    import utils as oasis_utils  # noqa: WPS433

    pkl_files = sorted(
        f for f in os.listdir(test_dir) if f.endswith(".pkl") and f.startswith("p_")
    )
    if not pkl_files:
        raise FileNotFoundError(f"No p_*.pkl under {test_dir}")

    with torch.no_grad():
        for name in pkl_files:
            path = os.path.join(test_dir, name)
            stem = os.path.splitext(name)[0]
            case_name = stem[2:] if stem.startswith("p_") else stem
            out_npz = os.path.join(out_root, f"disp_{case_name}.npz")
            if os.path.isfile(out_npz):
                print(f"skip existing {out_npz}", flush=True)
                continue

            pack = oasis_utils.pkload(path)
            if len(pack) == 4:
                x, y, _xs, _ys = pack
            else:
                x, y = pack[0], pack[1]

            x = np.ascontiguousarray(x[None, None, ...].astype(np.float32))
            y = np.ascontiguousarray(y[None, None, ...].astype(np.float32))
            xt = torch.from_numpy(x).to(device)
            yt = torch.from_numpy(y).to(device)

            _warped, flow = mod.forward(model, xt, yt)
            flow = flow.detach().float().cpu().numpy()[0]
            flow_hr = np.array([zoom(flow[i], 0.5, order=2) for i in range(3)]).astype(np.float16)
            np.savez(out_npz, flow_hr)
            print(f"wrote {out_npz} shape={flow_hr.shape}", flush=True)

    print(f"Done model_id={args.model_id} -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
