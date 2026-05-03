"""Pure-forward architecture profiling (random weights, no checkpoint needed).
TransMorphBayes: single forward pass (architecture speed, not MC-inference speed).
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TM   = os.path.join(REPO, "IXI", "TransMorph")
BLC  = os.path.join(REPO, "IXI", "Baseline_registration_methods")
BLT  = os.path.join(REPO, "IXI", "Baseline_Transformers")
SZ   = (160, 192, 224)
DEV  = "cuda" if torch.cuda.is_available() else "cpu"
WU, N = 5, 20

def _add(p):
    if p not in sys.path: sys.path.insert(0, p)

def _profile(fn, warmup=WU, n=N):
    with torch.no_grad():
        for _ in range(warmup): fn()
    if DEV == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter(); fn()
            if DEV == "cuda": torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    mem = torch.cuda.max_memory_allocated() / 1e9 if DEV == "cuda" else 0.0
    return round(float(np.mean(ts)), 4), round(float(np.std(ts, ddof=0)), 4), round(mem, 3)

def _n(m): return round(sum(p.numel() for p in m.parameters()) / 1e6, 3)

results = {}

# ── TransMorph / HypEReg-TransMorph (same architecture) ──
_add(TM)
try:
    from models.TransMorph import CONFIGS as C; import models.TransMorph as TM_mod
    m = TM_mod.TransMorph(C["TransMorph"]).to(DEV).eval()
    x = torch.randn(1, 2, *SZ, device=DEV)
    ms, sd, mem = _profile(lambda: m(x))
    results["HypEReg-TransMorph"] = {"mean_s": ms, "std_s": sd, "peak_gb": mem, "params_M": _n(m)}
    results["TransMorph"]          = {"mean_s": ms, "std_s": sd, "peak_gb": mem, "params_M": _n(m),
                                      "note": "same arch as HypEReg-TransMorph"}
    print(f"TransMorph family:  {ms:.4f}s  ±{sd:.4f}  {mem:.3f}GB  {_n(m)}M")
except Exception as e:
    results["TransMorph family"] = {"error": str(e)}
    print(f"TransMorph FAIL: {e}")

# ── TransMorphBayes (single pass) ──
try:
    from models.TransMorph_Bayes import CONFIGS as CB; import models.TransMorph_Bayes as TMB
    mb = TMB.TransMorphBayes(CB["TransMorphBayes"]).to(DEV).eval()
    xb = torch.randn(1, 2, *SZ, device=DEV)
    ms, sd, mem = _profile(lambda: mb(xb))
    results["TransMorphBayes"] = {"mean_s": ms, "std_s": sd, "peak_gb": mem, "params_M": _n(mb),
                                   "note": "single-pass; MC-inference = x25 runtime"}
    print(f"TransMorphBayes(1): {ms:.4f}s  ±{sd:.4f}  {mem:.3f}GB  {_n(mb)}M")
except Exception as e:
    results["TransMorphBayes"] = {"error": str(e)}
    print(f"TransMorphBayes FAIL: {e}")

# ── VoxelMorph-1 ──
try:
    import importlib, types
    for mod in list(sys.modules.keys()):
        if mod == "models" or mod.startswith("models."): del sys.modules[mod]
    _add(os.path.join(BLC, "VoxelMorph"))
    from models import VxmDense_1
    m = VxmDense_1(SZ).to(DEV).eval()
    x = torch.randn(1, 2, *SZ, device=DEV)
    ms, sd, mem = _profile(lambda: m(x))
    results["VoxelMorph-1"] = {"mean_s": ms, "std_s": sd, "peak_gb": mem, "params_M": _n(m)}
    print(f"VoxelMorph-1:       {ms:.4f}s  ±{sd:.4f}  {mem:.3f}GB  {_n(m)}M")
except Exception as e:
    results["VoxelMorph-1"] = {"error": str(e)}
    print(f"VoxelMorph-1 FAIL: {e}")

# ── CycleMorph ──
try:
    for mod in list(sys.modules.keys()):
        if mod == "models" or mod.startswith("models."): del sys.modules[mod]
    _add(os.path.join(BLC, "CycleMorph"))
    from models.networks import ResnetGenerator as CycleNet
    # CycleMorph registration network (3D version)
    m = CycleNet(input_nc=2, output_nc=3, ngf=32, n_downsampling=4, n_blocks=6,
                 img_shape=SZ).to(DEV).eval()
    x = torch.randn(1, 2, *SZ, device=DEV)
    ms, sd, mem = _profile(lambda: m(x))
    results["CycleMorph"] = {"mean_s": ms, "std_s": sd, "peak_gb": mem, "params_M": _n(m)}
    print(f"CycleMorph:         {ms:.4f}s  ±{sd:.4f}  {mem:.3f}GB  {_n(m)}M")
except Exception as e:
    results["CycleMorph"] = {"error": str(e)}
    print(f"CycleMorph FAIL: {e}")

# ── MIDIR ──
try:
    for mod in list(sys.modules.keys()):
        if mod == "models" or mod.startswith("models."): del sys.modules[mod]
    _add(os.path.join(BLC, "MIDIR"))
    from model import LocalModel
    m = LocalModel(SZ, int_steps=7).to(DEV).eval()
    x = torch.randn(1, 1, *SZ, device=DEV)
    ms, sd, mem = _profile(lambda: m(x, x))
    results["MIDIR"] = {"mean_s": ms, "std_s": sd, "peak_gb": mem, "params_M": _n(m)}
    print(f"MIDIR:              {ms:.4f}s  ±{sd:.4f}  {mem:.3f}GB  {_n(m)}M")
except Exception as e:
    results["MIDIR"] = {"error": str(e)}
    print(f"MIDIR FAIL: {e}")

# ── CoTr ──
try:
    for mod in list(sys.modules.keys()):
        if mod == "models" or mod.startswith("models."): del sys.modules[mod]
    _add(BLT); _add(os.path.join(BLT, "models"))
    from models.CoTr.network_architecture.ResTranUnet import ResTranUnet
    m = ResTranUnet(norm_cfg="IN", activation_cfg="relu",
                    num_classes=3, img_size=SZ).to(DEV).eval()
    x = torch.randn(1, 2, *SZ, device=DEV)
    ms, sd, mem = _profile(lambda: m(x))
    results["CoTr"] = {"mean_s": ms, "std_s": sd, "peak_gb": mem, "params_M": _n(m)}
    print(f"CoTr:               {ms:.4f}s  ±{sd:.4f}  {mem:.3f}GB  {_n(m)}M")
except Exception as e:
    results["CoTr"] = {"error": str(e)}
    print(f"CoTr FAIL: {e}")

# ── nnFormer ──
try:
    for mod in list(sys.modules.keys()):
        if mod == "models" or mod.startswith("models."): del sys.modules[mod]
    _add(BLT)
    from models.nnFormer.Swin_Unet_l_gelunorm import SwinUnet as NNF
    m = NNF(img_size=SZ[0]).to(DEV).eval()
    x = torch.randn(1, 2, *SZ, device=DEV)
    ms, sd, mem = _profile(lambda: m(x))
    results["nnFormer"] = {"mean_s": ms, "std_s": sd, "peak_gb": mem, "params_M": _n(m)}
    print(f"nnFormer:           {ms:.4f}s  ±{sd:.4f}  {mem:.3f}GB  {_n(m)}M")
except Exception as e:
    results["nnFormer"] = {"error": str(e)}
    print(f"nnFormer FAIL: {e}")

# ── PVT ──
try:
    for mod in list(sys.modules.keys()):
        if mod == "models" or mod.startswith("models."): del sys.modules[mod]
    _add(BLT)
    from models.PVT import PyramidVisionTransformerV2 as PVT
    m = PVT().to(DEV).eval()
    x = torch.randn(1, 2, *SZ, device=DEV)
    ms, sd, mem = _profile(lambda: m(x))
    results["PVT"] = {"mean_s": ms, "std_s": sd, "peak_gb": mem, "params_M": _n(m)}
    print(f"PVT:                {ms:.4f}s  ±{sd:.4f}  {mem:.3f}GB  {_n(m)}M")
except Exception as e:
    results["PVT"] = {"error": str(e)}
    print(f"PVT FAIL: {e}")

# ── SyN ──
results["SyN (ANTs)"] = {"mean_s": None, "peak_gb": None, "params_M": None,
                          "note": "classical iterative optimizer; no GPU forward pass"}
print("SyN (ANTs):         N/A (classical)")

# ── Save ──
out = os.path.join(REPO, "IXI", "Results", "profile_pure_forward.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nDone. Saved to {out}")
