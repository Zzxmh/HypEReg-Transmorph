# -*- coding: utf-8 -*-
import os
import sys

import torch

# Full-res PSC IXI per plan
CKPT_REL = os.path.join(
    "TransMorph_PSC", "experiments", "TransMorph_PSC_ixi", "p2a_end.pth.tar"
)


def _repo_root() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )


def build_model(device: str = "cuda"):
    root = _repo_root()
    psc = os.path.join(root, "TransMorph_PSC")
    if psc not in sys.path:
        sys.path.insert(0, psc)
    from models.TransMorph import CONFIGS as CONFIGS_TM
    import models.TransMorph as TransMorph  # noqa: WPS433

    ck = os.path.join(root, CKPT_REL)
    if not os.path.isfile(ck):
        raise FileNotFoundError(f"PSC checkpoint missing: {ck}")
    config = CONFIGS_TM["TransMorph-fullres"]
    model = TransMorph.TransMorph(config)
    print(f"TransMorph+PSC: loading {ck}", flush=True)
    z = torch.load(ck, map_location="cpu", weights_only=False)
    sd = z["state_dict"] if "state_dict" in z else z
    model.load_state_dict(sd)
    return model.to(device), config


def forward(model, x, y) -> tuple:
    x_in = torch.cat((x, y), dim=1)
    return model(x_in)
