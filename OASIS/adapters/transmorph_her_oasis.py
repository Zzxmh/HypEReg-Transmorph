# -*- coding: utf-8 -*-
"""HypEReg-TransMorph trained on OASIS (same loss weights as IXI HypEReg run)."""
from __future__ import annotations

import os
import sys

import torch

from ._helpers import add_sys_path, latest_ckpt_in_dir, load_state_dict_any, oasis_transmorph_dir


HER_DIR = "TransMorph_OASIS_HER_ncc_1.0_grad_1.0_her_1.0_a0_b0.02_g20"


def _ckpt_path() -> str:
    d = os.path.join(oasis_transmorph_dir(), "experiments", HER_DIR)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Missing HypEReg experiment dir: {d}")
    return latest_ckpt_in_dir(d)


def build_model(device: str = "cuda"):
    tdir = oasis_transmorph_dir()
    add_sys_path(tdir)
    from models.TransMorph import CONFIGS as CONFIGS_TM
    import models.TransMorph as TransMorph  # noqa: WPS433

    ck = _ckpt_path()
    config = CONFIGS_TM["TransMorph"]
    model = TransMorph.TransMorph(config)
    print(f"transmorph_her_oasis: loading {ck}", flush=True)
    model.load_state_dict(load_state_dict_any(ck))
    return model.to(device), config


def forward(model, x, y):
    x_in = torch.cat((x, y), dim=1)
    return model(x_in)
