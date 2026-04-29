# -*- coding: utf-8 -*-
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
    return os.path.join(
        _base_dir(), "models", "nnFormer", "nnFormer_Validation_dsc0.739.pth.tar"
    )


def build_model(device: str = "cuda"):
    base = _base_dir()
    purge_module_prefix("models")
    purge_module_prefix("nnFormer")
    ensure_namespace_package("models", os.path.join(base, "models"))
    add_sys_path(os.path.join(base, "models"))
    from nnFormer.Swin_Unet_l_gelunorm import (  # noqa: WPS433
        swintransformer as nnFormer,
    )

    ckpt = _ckpt_path()
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"nnFormer ckpt missing: {ckpt}")
    model = nnFormer()
    model.load_state_dict(load_state_dict_any(ckpt))
    return model.to(device), cfg_with_size((160, 192, 224))


def forward(model, x, y):
    x_in = torch.cat((x, y), dim=1)
    out = model(x_in)
    return out[0], out[1]
