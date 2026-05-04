# -*- coding: utf-8 -*-
"""TransMorph semi-supervised (DSC) checkpoint released for OASIS / Learn2Reg."""
from __future__ import annotations

import os
import sys

import torch

from ._helpers import add_sys_path, load_state_dict_any, oasis_transmorph_dir


def _ckpt_path() -> str:
    p = os.path.join(oasis_transmorph_dir(), "TransMorph_Validation_dsc0.857.pth.tar")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing OASIS TransMorph ckpt: {p}")
    return p


def build_model(device: str = "cuda"):
    tdir = oasis_transmorph_dir()
    add_sys_path(tdir)
    from models.TransMorph import CONFIGS as CONFIGS_TM
    import models.TransMorph as TransMorph  # noqa: WPS433

    config = CONFIGS_TM["TransMorph"]
    model = TransMorph.TransMorph(config)
    ck = _ckpt_path()
    print(f"transmorph_dsc857_oasis: loading {ck}", flush=True)
    model.load_state_dict(load_state_dict_any(ck))
    return model.to(device), config


def forward(model, x, y):
    x_in = torch.cat((x, y), dim=1)
    return model(x_in)
