# -*- coding: utf-8 -*-
import glob
import os
import sys

import torch
from natsort import natsorted

def _transmorph_dir() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "TransMorph")
    )


def _ckpt_path() -> str:
    d = _transmorph_dir()
    uploaded = os.path.join(d, "experiments", "TransMorph_Validation_dsc0.744.pth.tar")
    if os.path.isfile(uploaded):
        return uploaded
    ex = os.path.join(d, "experiments", "TransMorph_IXI_ncc_1.0_grad_1.0")
    files = natsorted(glob.glob(os.path.join(ex, "*.pth*")))
    if not files:
        raise FileNotFoundError(f"No checkpoint under {ex} or uploaded file {uploaded}")
    return files[-1]


def build_model(device: str = "cuda"):
    tdir = _transmorph_dir()
    if tdir not in sys.path:
        sys.path.insert(0, tdir)
    from models.TransMorph import CONFIGS as CONFIGS_TM
    import models.TransMorph as TransMorph  # noqa: WPS433

    config = CONFIGS_TM["TransMorph"]
    model = TransMorph.TransMorph(config)
    print(f"TransMorph: loading { _ckpt_path() }", flush=True)
    ck = _ckpt_path()
    z = torch.load(ck, map_location="cpu", weights_only=False)
    sd = z["state_dict"] if isinstance(z, dict) and "state_dict" in z else z
    model.load_state_dict(sd)
    return model.to(device), config


def forward(model, x, y) -> tuple:
    """x, y: (B,1,D,H,W). Returns (warped_moving, flow)."""
    x_in = torch.cat((x, y), dim=1)
    return model(x_in)
