from __future__ import annotations

import glob
import os
import sys
import types
from types import SimpleNamespace

import torch


def repo_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def oasis_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def ixi_root() -> str:
    return os.path.normpath(os.path.join(repo_root(), "IXI"))


def oasis_transmorph_dir() -> str:
    return os.path.join(oasis_root(), "TransMorph")


def add_sys_path(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


def ensure_namespace_package(name: str, path: str) -> None:
    mod = types.ModuleType(name)
    mod.__path__ = [path]  # type: ignore[attr-defined]
    sys.modules[name] = mod


def purge_module_prefix(prefix: str) -> None:
    drop = [k for k in sys.modules if k == prefix or k.startswith(prefix + ".")]
    for k in drop:
        sys.modules.pop(k, None)


def load_state_dict_any(ckpt_path: str):
    z = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return z["state_dict"] if isinstance(z, dict) and "state_dict" in z else z


def cfg_with_size(size):
    return SimpleNamespace(img_size=tuple(size))


def latest_ckpt_in_dir(exp_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(exp_dir, "*.pth*")))
    if not files:
        raise FileNotFoundError(f"No checkpoints under {exp_dir}")
    return files[-1]
