# -*- coding: utf-8 -*-
"""TransMorph (IXI-trained) — zero-shot transfer to OASIS.

Loads the IXI validation checkpoint (dsc0.744) without any OASIS fine-tuning,
enabling a direct cross-cohort generalisation measurement.
"""
from __future__ import annotations

import glob
import os

import torch

from ._helpers import add_sys_path, ixi_root, load_state_dict_any, purge_module_prefix


def _ixi_transmorph_dir() -> str:
    return os.path.join(ixi_root(), "TransMorph")


def _ckpt_path() -> str:
    uploaded = os.path.join(
        _ixi_transmorph_dir(), "experiments", "TransMorph_Validation_dsc0.744.pth.tar"
    )
    if os.path.isfile(uploaded):
        return uploaded
    ex = os.path.join(_ixi_transmorph_dir(), "experiments", "TransMorph_IXI_ncc_1.0_grad_1.0")
    files = sorted(glob.glob(os.path.join(ex, "*.pth*")))
    if not files:
        raise FileNotFoundError(
            f"transmorph_zs_oasis: no checkpoint at {uploaded} or under {ex}"
        )
    return files[-1]


def build_model(device: str = "cuda"):
    tdir = _ixi_transmorph_dir()
    purge_module_prefix("models")
    add_sys_path(tdir)
    from models.TransMorph import CONFIGS as CONFIGS_TM
    import models.TransMorph as TransMorph  # noqa: WPS433

    config = CONFIGS_TM["TransMorph"]
    model = TransMorph.TransMorph(config)
    ck = _ckpt_path()
    print(f"transmorph_zs_oasis: loading IXI ckpt {ck}", flush=True)
    model.load_state_dict(load_state_dict_any(ck))
    return model.to(device), config


def forward(model, x, y):
    x_in = torch.cat((x, y), dim=1)
    return model(x_in)
