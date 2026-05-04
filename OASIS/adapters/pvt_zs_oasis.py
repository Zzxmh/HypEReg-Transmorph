# -*- coding: utf-8 -*-
"""PVT (IXI-trained) — zero-shot transfer to OASIS."""
from __future__ import annotations

import os

import torch

from ._helpers import (
    add_sys_path,
    cfg_with_size,
    ensure_namespace_package,
    ixi_root,
    load_state_dict_any,
    purge_module_prefix,
)


def _base_dir() -> str:
    return os.path.join(ixi_root(), "Baseline_Transformers")


def _ckpt_path() -> str:
    return os.path.join(_base_dir(), "PVT_Validation_dsc0.720.pth.tar")


def build_model(device: str = "cuda"):
    base = _base_dir()
    purge_module_prefix("models")
    purge_module_prefix("PVT")
    ensure_namespace_package("models", os.path.join(base, "models"))
    add_sys_path(os.path.join(base, "models"))
    from PVT import CONFIGS, PVTVNetSkip  # noqa: WPS433

    ckpt = _ckpt_path()
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"pvt_zs_oasis: PVT IXI ckpt missing: {ckpt}")
    config_pvt = CONFIGS["PVT-Net"]
    model = PVTVNetSkip(config_pvt)
    model.load_state_dict(load_state_dict_any(ckpt))
    print(f"pvt_zs_oasis: loading IXI ckpt {ckpt}", flush=True)
    return model.to(device), config_pvt


def forward(model, x, y):
    x_in = torch.cat((x, y), dim=1)
    out = model(x_in)
    return out[0], out[1]
