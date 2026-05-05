# -*- coding: utf-8 -*-
"""HypEReg-TransMorph (IXI-trained) — zero-shot transfer to OASIS.

Loads the IXI HypEReg checkpoint (dsc0.743) trained with HypEReg loss
(alpha=0, beta=0.02, gamma=20) without any OASIS fine-tuning.
This is the cross-cohort generalisation condition for the HypEReg paper.
"""
from __future__ import annotations

import os

import torch

from ._helpers import add_sys_path, ixi_root, latest_ckpt_in_dir, load_state_dict_any, purge_module_prefix


HER_IXI_EXPERIMENT = "TransMorph_IXI_HER_ncc_1.0_grad_1.0_her_1.0_a0_b0.02_g20"
HER_IXI_CKPT_NAME = "dsc0.743.pth.tar"


def _ixi_transmorph_dir() -> str:
    return os.path.join(ixi_root(), "TransMorph")


def _ckpt_path() -> str:
    tdir = _ixi_transmorph_dir()
    uploaded = os.path.join(tdir, "experiments", HER_IXI_EXPERIMENT, HER_IXI_CKPT_NAME)
    if os.path.isfile(uploaded):
        return uploaded
    exp_dir = os.path.join(tdir, "experiments", HER_IXI_EXPERIMENT)
    if os.path.isdir(exp_dir):
        return latest_ckpt_in_dir(exp_dir)
    raise FileNotFoundError(
        f"transmorph_her_zs_oasis: IXI HypEReg experiment dir missing: {exp_dir}"
    )


def build_model(device: str = "cuda"):
    tdir = _ixi_transmorph_dir()
    purge_module_prefix("models")
    add_sys_path(tdir)
    from models.TransMorph import CONFIGS as CONFIGS_TM
    import models.TransMorph as TransMorph  # noqa: WPS433

    config = CONFIGS_TM["TransMorph"]
    model = TransMorph.TransMorph(config)
    ck = _ckpt_path()
    print(f"transmorph_her_zs_oasis: loading IXI HypEReg ckpt {ck}", flush=True)
    model.load_state_dict(load_state_dict_any(ck))
    return model.to(device), config


def forward(model, x, y):
    x_in = torch.cat((x, y), dim=1)
    return model(x_in)
