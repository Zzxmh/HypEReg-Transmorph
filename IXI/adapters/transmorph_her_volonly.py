# -*- coding: utf-8 -*-
import os
import sys

import torch

CKPT_GLOB_DIR = "TransMorph_IXI_HER_ncc_1.0_grad_1.0_her_1.0_a0_b0.02_g0.0"


def _transmorph_dir() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "TransMorph"))


def _ckpt_path() -> str:
    tdir = _transmorph_dir()
    ex = os.path.join(tdir, "experiments", CKPT_GLOB_DIR)
    if not os.path.isdir(ex):
        raise FileNotFoundError(f"Experiment dir missing: {ex}")
    cands = [
        f for f in os.listdir(ex)
        if f.endswith((".pth.tar", ".pth", ".tar")) and os.path.isfile(os.path.join(ex, f))
    ]
    if not cands:
        raise FileNotFoundError(f"No checkpoint files under {ex}")
    # keep same selection behavior as other adapters: latest by lexicographic
    cands = sorted(cands)
    return os.path.join(ex, cands[-1])


def build_model(device: str = "cuda"):
    tdir = _transmorph_dir()
    if tdir not in sys.path:
        sys.path.insert(0, tdir)
    from models.TransMorph import CONFIGS as CONFIGS_TM
    import models.TransMorph as TransMorph  # noqa: WPS433

    ck = _ckpt_path()
    config = CONFIGS_TM["TransMorph"]
    model = TransMorph.TransMorph(config)
    print(f"HypEReg-TransMorph (volume-only): loading {ck}", flush=True)
    z = torch.load(ck, map_location="cpu", weights_only=False)
    sd = z["state_dict"] if isinstance(z, dict) and "state_dict" in z else z
    model.load_state_dict(sd)
    return model.to(device), config


def forward(model, x, y) -> tuple:
    x_in = torch.cat((x, y), dim=1)
    return model(x_in)

