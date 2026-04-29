# -*- coding: utf-8 -*-
"""Lightweight IXI eval for uploaded checkpoints (Dice + non_jec only)."""
from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

IXI_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(IXI_DIR)
if IXI_DIR not in sys.path:
    sys.path.insert(0, IXI_DIR)


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _load_cfg(path: str) -> Dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_model(cfg: Dict, model_id: str) -> Dict:
    for m in cfg.get("models", []):
        if m.get("id") == model_id:
            return m
    raise KeyError(f"Model id not found in config: {model_id}")


def run_one(model_entry: Dict, cfg: Dict, out_dir: str) -> str:
    from TransMorph import utils as tm_utils  # noqa: WPS433
    from TransMorph.data import datasets, trans  # noqa: WPS433
    from metrics_full import jacobian_stats, per_label_dice  # noqa: WPS433

    model_id = model_entry.get("id")
    adapter_name = model_entry.get("adapter", model_id)
    ad = importlib.import_module(f"adapters.{adapter_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = ad.build_model(device=device)
    model.eval()
    reg = tm_utils.register_model(config.img_size, "bilinear")
    if device == "cuda":
        reg.cuda()

    data_root = _abs(cfg.get("ixi_root", "IXI_data"))
    test_subdir = cfg.get("test_subdir", "Test")
    test_dir = os.path.join(data_root, test_subdir)
    pkl_list = sorted([os.path.join(test_dir, p) for p in os.listdir(test_dir) if p.endswith(".pkl")])
    atlas = cfg.get("atlas")
    atlas = _abs(atlas) if atlas else os.path.join(data_root, "atlas.pkl")
    if not os.path.isfile(atlas):
        atlas = os.path.join(data_root, "altas.pkl")
    if not os.path.isfile(atlas):
        raise FileNotFoundError(f"Atlas missing under {data_root}")

    num_lbl = int(cfg.get("num_labels", 46))
    test_tf = transforms.Compose([trans.Seg_norm(), trans.NumpyType((np.float32, np.int16))])
    ds = datasets.IXIBrainInferDataset(pkl_list, atlas, transforms=test_tf)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=(device == "cuda"), drop_last=True)

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{model_id}_light_per_case.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pkl", "dice_mean", "non_jec", "inference_s"])
        w.writeheader()
        for i, data in enumerate(dl):
            x, y, x_seg, y_seg = [t.to(device) for t in data]
            t0 = time.time()
            with torch.no_grad():
                x_def, flow = ad.forward(model, x, y)
            inf_s = time.time() - t0

            # label warp for Dice
            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=num_lbl)
            x_seg_oh = torch.squeeze(x_seg_oh, 1).permute(0, 4, 1, 2, 3).contiguous()
            x_segs: List[torch.Tensor] = []
            for c in range(num_lbl):
                def_seg = reg([x_seg_oh[:, c : c + 1, ...].float(), flow.float()])
                x_segs.append(def_seg)
            pred = torch.argmax(torch.cat(x_segs, dim=1), dim=1, keepdim=True).long().cpu().numpy()[0, 0]
            true = y_seg.long().cpu().numpy()[0, 0]
            dices, _ = per_label_dice(pred, true, num_classes=num_lbl)
            d_mean = float(np.mean(dices))
            non_j = float(jacobian_stats(flow.detach().cpu().numpy()[0], n_bins=100)["non_jec"])
            sid = os.path.splitext(os.path.basename(pkl_list[i]))[0]
            w.writerow({"pkl": sid, "dice_mean": d_mean, "non_jec": non_j, "inference_s": float(inf_s)})
            if (i + 1) % 10 == 0 or (i + 1) == len(dl):
                print(f"[light-eval] {model_id} {i + 1}/{len(dl)}", flush=True)
    return out_csv


def main() -> int:
    ap = argparse.ArgumentParser(description="Light eval (Dice + non_jec) for specified models.")
    ap.add_argument("--config", default=os.path.join(IXI_DIR, "eval_configs.yaml"))
    ap.add_argument("--models", default="transmorph_original,transmorphbayes")
    ap.add_argument("--out_dir", default=os.path.join("IXI", "Results", "uploaded_weights_light"))
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    out_dir = _abs(args.out_dir)
    mids = [m.strip() for m in args.models.split(",") if m.strip()]
    for mid in mids:
        me = _find_model(cfg, mid)
        path = run_one(me, cfg, out_dir)
        print(f"Wrote: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

