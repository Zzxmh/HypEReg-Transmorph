"""
Legacy HER OASIS submission exporter (same half-res ``disp_*.npz`` as ``export_displacements.py``).

Writes under ``OASIS/data/Submit/submission/{model_id}/task_03/`` so ``eval_oasis.py`` can find predictions.

Usage (from repository root)::

    python OASIS/TransMorph/submit_TransMorph_her.py --model-id transmorph_her_oasis
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
from natsort import natsorted
from scipy.ndimage import zoom


def main() -> None:
    ap = argparse.ArgumentParser(description="Export OASIS HER TransMorph disp npz files.")
    ap.add_argument(
        "--model-id",
        default="transmorph_her_oasis",
        help="Subfolder name under OASIS/data/Submit/submission/<model_id>/task_03/",
    )
    args = ap.parse_args()

    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    tm_root = os.path.join(repo_root, "OASIS", "TransMorph")
    if tm_root not in sys.path:
        sys.path.insert(0, tm_root)

    import utils  # noqa: WPS433
    from models.TransMorph import CONFIGS as CONFIGS_TM
    import models.TransMorph as TransMorph  # noqa: WPS433

    test_dir = os.path.join(repo_root, "OASIS", "data", "Test")
    save_dir = os.path.join(
        repo_root, "OASIS", "data", "Submit", "submission", args.model_id, "task_03"
    )
    os.makedirs(save_dir, exist_ok=True)

    model_folder = "TransMorph_OASIS_HER_ncc_1.0_grad_1.0_her_1.0_a0_b0.02_g20"
    exp_dir = os.path.join(tm_root, "experiments", model_folder)
    ckpts = natsorted(glob.glob(os.path.join(exp_dir, "*.pth*")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints under {exp_dir}")
    best_path = ckpts[-1]
    print("Best model:", os.path.basename(best_path), flush=True)

    config = CONFIGS_TM["TransMorph"]
    model = TransMorph.TransMorph(config)
    z = torch.load(best_path, map_location="cpu", weights_only=False)
    state = z["state_dict"] if isinstance(z, dict) and "state_dict" in z else z
    model.load_state_dict(state, strict=True)
    model.cuda()
    model.eval()

    file_names = natsorted(glob.glob(os.path.join(test_dir, "*.pkl")))
    with torch.no_grad():
        for data in file_names:
            x, y, _, _ = utils.pkload(data)
            x, y = x[None, None, ...], y[None, None, ...]
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)
            x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
            file_name = os.path.basename(data).split(".")[0][2:]
            print(file_name, flush=True)
            x_in = torch.cat((x, y), dim=1)
            _, flow = model(x_in)
            flow = flow.cpu().detach().numpy()[0]
            flow = np.array([zoom(flow[i], 0.5, order=2) for i in range(3)]).astype(np.float16)
            print(flow.shape, flush=True)
            np.savez(os.path.join(save_dir, f"disp_{file_name}.npz"), flow)


if __name__ == "__main__":
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print("Number of GPU: " + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print("     GPU #" + str(GPU_idx) + ": " + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print("Currently using: " + torch.cuda.get_device_name(GPU_iden))
    print("If the GPU is available? " + str(GPU_avai))
    main()
