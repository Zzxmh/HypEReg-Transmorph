from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import torch


def repo_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def ixi_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def add_sys_path(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


def purge_module_prefix(prefix: str) -> None:
    drop = [k for k in sys.modules.keys() if k == prefix or k.startswith(prefix + ".")]
    for k in drop:
        sys.modules.pop(k, None)


def ensure_namespace_package(name: str, path: str) -> None:
    mod = types.ModuleType(name)
    mod.__path__ = [path]  # type: ignore[attr-defined]
    sys.modules[name] = mod


def load_state_dict_any(ckpt_path: str):
    z = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return z["state_dict"] if isinstance(z, dict) and "state_dict" in z else z


def cfg_with_size(size):
    return SimpleNamespace(img_size=tuple(size))
