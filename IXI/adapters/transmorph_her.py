# -*- coding: utf-8 -*-
import os
import sys

import torch

CKPT_NAME = "dsc0.743.pth.tar"
HER_EXPERIMENT = "TransMorph_IXI_HER_ncc_1.0_grad_1.0_her_1.0_a0_b0.02_g20"


def _transmorph_dir() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "TransMorph")
    )


def build_model(device: str = "cuda"):
    tdir = _transmorph_dir()
    if tdir not in sys.path:
        sys.path.insert(0, tdir)
    from models.TransMorph import CONFIGS as CONFIGS_TM
    import models.TransMorph as TransMorph  # noqa: WPS433

    p = os.path.join(tdir, "experiments", HER_EXPERIMENT, CKPT_NAME)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"HypEReg checkpoint missing: {p}")
    config = CONFIGS_TM["TransMorph"]
    model = TransMorph.TransMorph(config)
    print(f"HypEReg-TransMorph: loading {p}", flush=True)
    sd = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd)
    return model.to(device), config


def forward(model, x, y) -> tuple:
    x_in = torch.cat((x, y), dim=1)
    return model(x_in)
