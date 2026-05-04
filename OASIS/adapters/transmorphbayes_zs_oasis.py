# -*- coding: utf-8 -*-
"""TransMorphBayes (IXI-trained) — zero-shot transfer to OASIS.

Uses 25-sample MC-dropout inference to obtain the representative flow,
identical to the IXI evaluation protocol.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch

from ._helpers import add_sys_path, ixi_root, load_state_dict_any, purge_module_prefix


def _ixi_transmorph_dir() -> str:
    return os.path.join(ixi_root(), "TransMorph")


def _ckpt_path() -> str:
    uploaded = os.path.join(
        _ixi_transmorph_dir(),
        "experiments",
        "TransMorph_Bayes_Validation_dsc0.743.pth.tar",
    )
    if os.path.isfile(uploaded):
        return uploaded
    ex = os.path.join(
        _ixi_transmorph_dir(), "experiments", "TransMorphBayes_ncc_1_diffusion_1"
    )
    files = sorted(glob.glob(os.path.join(ex, "*.pth*")))
    if not files:
        raise FileNotFoundError(
            f"transmorphbayes_zs_oasis: no checkpoint at {uploaded} or under {ex}"
        )
    return files[-1]


def build_model(device: str = "cuda"):
    tdir = _ixi_transmorph_dir()
    purge_module_prefix("models")
    add_sys_path(tdir)
    from models.TransMorph_Bayes import CONFIGS as CONFIGS_TM
    import models.TransMorph_Bayes as TransMorph_Bayes  # noqa: WPS433

    config = CONFIGS_TM["TransMorphBayes"]
    model = TransMorph_Bayes.TransMorphBayes(config)
    ck = _ckpt_path()
    print(f"transmorphbayes_zs_oasis: loading IXI ckpt {ck}", flush=True)
    model.load_state_dict(load_state_dict_any(ck))
    return model.to(device), config


def _mc_preds(model, x_in, mc_iter: int = 25):
    outputs, flows = [], []
    model.train()
    with torch.no_grad():
        for _ in range(mc_iter):
            out, flow = model(x_in)
            outputs.append(out)
            flows.append(flow)
    return outputs, flows


def forward(model, x, y):
    x_in = torch.cat((x, y), dim=1)
    outputs, flows = _mc_preds(model, x_in, mc_iter=25)
    outs_np = [o.detach().cpu().numpy() for o in outputs]
    mean_np = np.mean(np.stack(outs_np, axis=0), axis=0)
    errs = [float(np.mean((o - mean_np) ** 2)) for o in outs_np]
    idx = int(np.argmin(np.asarray(errs, dtype=np.float64)))
    return outputs[idx], flows[idx]
