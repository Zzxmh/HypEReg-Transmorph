# -*- coding: utf-8 -*-
"""
TransMorph without HypEReg on OASIS.

Prefers an OASIS-retrained unsupervised experiment directory under
``OASIS/TransMorph/experiments/`` (any ``TransMorph_*`` folder that does not
contain ``HypEReg``). If none exists, falls back to the IXI TransMorph validation
checkpoint (same architecture) for zero-shot OASIS evaluation; this is flagged
in stdout so the manuscript can distinguish true OASIS-retrained vs fallback.
"""
from __future__ import annotations

import glob
import os
import sys

import torch

from ._helpers import add_sys_path, ixi_root, latest_ckpt_in_dir, load_state_dict_any, oasis_transmorph_dir


def _oasis_unsup_ckpt() -> str | None:
    exp_root = os.path.join(oasis_transmorph_dir(), "experiments")
    if not os.path.isdir(exp_root):
        return None
    candidates: list[str] = []
    for name in sorted(os.listdir(exp_root)):
        # Skip HypEReg experiments; checkpoint folder names may contain "HER" or "HypEReg"
        if "HER" in name.upper() or "HYPEREG" in name.upper():
            continue
        sub = os.path.join(exp_root, name)
        if not os.path.isdir(sub):
            continue
        try:
            candidates.append(latest_ckpt_in_dir(sub))
        except FileNotFoundError:
            continue
    return candidates[-1] if candidates else None


def _ixi_fallback_ckpt() -> str:
    p = os.path.join(ixi_root(), "TransMorph", "experiments", "TransMorph_Validation_dsc0.744.pth.tar")
    if os.path.isfile(p):
        return p
    ex = os.path.join(ixi_root(), "TransMorph", "experiments", "TransMorph_IXI_ncc_1.0_grad_1.0")
    if os.path.isdir(ex):
        files = sorted(glob.glob(os.path.join(ex, "*.pth*")))
        if files:
            return files[-1]
    raise FileNotFoundError(
        "No OASIS unsupervised TransMorph experiment and no IXI TransMorph_Validation_dsc0.744.pth.tar"
    )


def _ckpt_path() -> str:
    o = _oasis_unsup_ckpt()
    if o:
        return o
    fb = _ixi_fallback_ckpt()
    print(
        "transmorph_unsup_oasis: WARNING using IXI-trained TransMorph checkpoint for zero-shot OASIS eval:",
        fb,
        flush=True,
    )
    return fb


def build_model(device: str = "cuda"):
    tdir = oasis_transmorph_dir()
    add_sys_path(tdir)
    from models.TransMorph import CONFIGS as CONFIGS_TM
    import models.TransMorph as TransMorph  # noqa: WPS433

    config = CONFIGS_TM["TransMorph"]
    model = TransMorph.TransMorph(config)
    ck = _ckpt_path()
    print(f"transmorph_unsup_oasis: loading {ck}", flush=True)
    model.load_state_dict(load_state_dict_any(ck), strict=False)
    return model.to(device), config


def forward(model, x, y):
    x_in = torch.cat((x, y), dim=1)
    return model(x_in)
