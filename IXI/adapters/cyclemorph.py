# -*- coding: utf-8 -*-
from __future__ import annotations

import os

import torch

from ._helpers import (
    add_sys_path,
    cfg_with_size,
    ixi_root,
    load_state_dict_any,
    purge_module_prefix,
)


def _base_dir() -> str:
    return os.path.join(ixi_root(), "Baseline_registration_methods", "CycleMorph")


def _ckpt_path() -> str:
    return os.path.join(_base_dir(), "CycleMorph_Validation_dsc0.729.pth.tar")


def build_model(device: str = "cuda"):
    base = _base_dir()
    purge_module_prefix("models")
    purge_module_prefix("util")
    add_sys_path(base)
    from models.cycleMorph_model import CONFIGS, cycleMorph  # noqa: WPS433

    ckpt = _ckpt_path()
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"CycleMorph ckpt missing: {ckpt}")
    opt = CONFIGS["Cycle-Morph-v0"]
    model = cycleMorph()
    model.initialize(opt)
    net = model.netG_A
    net.load_state_dict(load_state_dict_any(ckpt))
    return net.to(device), cfg_with_size((160, 192, 224))


def forward(model, x, y):
    x_in = torch.cat((x, y), dim=1)
    return model(x_in)
