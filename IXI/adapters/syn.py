# -*- coding: utf-8 -*-
from __future__ import annotations

from types import SimpleNamespace

import ants
import numpy as np
import torch

from ._helpers import cfg_with_size

PREFERRED_DEVICE = "cpu"


class _ClassicalModel:
    def to(self, _device):
        return self

    def eval(self):
        return self


def _flow_from_ants_warp(warp_path: str) -> np.ndarray:
    arr = ants.image_read(warp_path).numpy()
    # Common ANTs vector field layouts:
    # - (D, H, W, 3)
    # - (D, H, W, 1, 3)
    if arr.ndim == 5:
        arr = arr[..., 0, :]
    if arr.shape[-1] != 3:
        raise RuntimeError(f"Unexpected SyN warp shape: {arr.shape}")
    return np.moveaxis(arr.astype(np.float32), -1, 0)


def build_model(device: str = "cuda"):
    _ = device
    return _ClassicalModel(), cfg_with_size((160, 192, 224))


def forward(model, x, y):
    _ = model
    moving = x.detach().cpu().numpy()[0, 0].astype(np.float32)
    fixed = y.detach().cpu().numpy()[0, 0].astype(np.float32)
    moving_a = ants.from_numpy(moving)
    fixed_a = ants.from_numpy(fixed)

    reg = ants.registration(
        fixed=fixed_a,
        moving=moving_a,
        type_of_transform="SyNOnly",
        reg_iterations=(160, 80, 40),
        syn_metric="meansquares",
    )
    warped_a = ants.apply_transforms(
        fixed=fixed_a, moving=moving_a, transformlist=reg["fwdtransforms"]
    )
    flow = _flow_from_ants_warp(reg["fwdtransforms"][0])

    warped_t = torch.from_numpy(warped_a.numpy()[None, None]).to(x.device)
    flow_t = torch.from_numpy(flow[None]).to(x.device)
    return warped_t, flow_t
