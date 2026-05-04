# -*- coding: utf-8 -*-
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from ._helpers import add_sys_path, cfg_with_size, ixi_root, load_state_dict_any, oasis_root, purge_module_prefix


def _base_dir() -> str:
    o = os.path.join(oasis_root(), "Baseline_registration_methods", "MIDIR")
    if os.path.isdir(o) and os.path.isfile(os.path.join(o, "MIDIR_Validation_dsc0.733.pth.tar")):
        return o
    return os.path.join(ixi_root(), "Baseline_registration_methods", "MIDIR")


def _ckpt_path() -> str:
    return os.path.join(_base_dir(), "MIDIR_Validation_dsc0.733.pth.tar")


def build_model(device: str = "cuda"):
    base = _base_dir()
    purge_module_prefix("models")
    purge_module_prefix("transformation")
    add_sys_path(base)
    import models as midir_models  # noqa: WPS433

    ckpt = _ckpt_path()
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"MIDIR ckpt missing: {ckpt}")
    model = midir_models.CubicBSplineNet(ndim=3)
    model.load_state_dict(load_state_dict_any(ckpt))
    print(f"midir_oasis: loading {ckpt} (base={base})", flush=True)
    return model.to(device), cfg_with_size((160, 192, 224))


def forward(model, x, y):
    x_def, _flow_lowres, disp = model((x, y))
    flow = F.interpolate(disp, size=x.shape[2:], mode="trilinear", align_corners=True)
    return x_def, flow
