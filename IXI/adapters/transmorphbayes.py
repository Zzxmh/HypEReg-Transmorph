# -*- coding: utf-8 -*-
import glob
import os
import sys

import numpy as np
import torch
from natsort import natsorted


def _transmorph_dir() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "TransMorph")
    )


def _ckpt_path() -> str:
    d = _transmorph_dir()
    uploaded = os.path.join(d, "experiments", "TransMorph_Bayes_Validation_dsc0.743.pth.tar")
    if os.path.isfile(uploaded):
        return uploaded
    ex = os.path.join(d, "experiments", "TransMorphBayes_ncc_1_diffusion_1")
    files = natsorted(glob.glob(os.path.join(ex, "*.pth*")))
    if not files:
        raise FileNotFoundError(f"No checkpoint under {ex} or uploaded file {uploaded}")
    return files[-1]


def build_model(device: str = "cuda"):
    tdir = _transmorph_dir()
    if tdir not in sys.path:
        sys.path.insert(0, tdir)
    from models.TransMorph_Bayes import CONFIGS as CONFIGS_TM
    import models.TransMorph_Bayes as TransMorph_Bayes  # noqa: WPS433

    config = CONFIGS_TM["TransMorphBayes"]
    model = TransMorph_Bayes.TransMorphBayes(config)
    ck = _ckpt_path()
    print(f"TransMorphBayes: loading {ck}", flush=True)
    z = torch.load(ck, map_location="cpu", weights_only=False)
    sd = z["state_dict"] if isinstance(z, dict) and "state_dict" in z else z
    model.load_state_dict(sd)
    return model.to(device), config


def _mc_preds(model, x_in, mc_iter: int = 25):
    outputs = []
    flows = []
    model.train()  # keep dropout active during MC sampling
    with torch.no_grad():
        for _ in range(int(mc_iter)):
            out, flow = model(x_in)
            outputs.append(out)
            flows.append(flow)
    return outputs, flows


def forward(model, x, y):
    """x, y: (B,1,D,H,W). Returns (warped_moving, flow)."""
    x_in = torch.cat((x, y), dim=1)
    outputs, flows = _mc_preds(model, x_in, mc_iter=25)
    # choose sample closest to MC mean prediction (same rule as legacy infer script)
    outs_np = [o.detach().cpu().numpy() for o in outputs]
    mean_np = np.mean(np.stack(outs_np, axis=0), axis=0)
    errs = [float(np.mean((o - mean_np) ** 2)) for o in outs_np]
    idx = int(np.argmin(np.asarray(errs, dtype=np.float64)))
    return outputs[idx], flows[idx]

