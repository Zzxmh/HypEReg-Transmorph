"""
Model registry and paths for comprehensive IXI analysis.
All checkpoint dirs are under IXI/TransMorph/ (repo-relative).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

# Repo root: IXI/../..
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IXI_DIR = os.path.join(_REPO_ROOT, "IXI")
TRANSMORPH_DIR = os.path.join(IXI_DIR, "TransMorph")
DATA_ROOT_DEFAULT = os.path.join(_REPO_ROOT, "IXI_data")
ATLAS_PKL = os.path.join(DATA_ROOT_DEFAULT, "atlas.pkl")
TEST_PKL_GLOB = os.path.join(DATA_ROOT_DEFAULT, "Test", "*.pkl")


@dataclass(frozen=True)
class ModelSpec:
    """One trained model to evaluate."""

    name: str
    ckpt_subdir: str
    """Path relative to IXI/TransMorph/ pointing at a folder containing .pth.tar."""
    backbone: str
    """'TransMorph' or 'TransMorphBayes'."""
    mc_iter: int = 25
    """MC dropout forward passes for Bayes; ignored otherwise."""
    ckpt_file: Optional[str] = None
    """If set, load this filename inside the ckpt_dir (e.g. dsc0.743.pth.tar)."""


# Six local models with on-disk weights (per plan)
MODEL_REGISTRY: List[ModelSpec] = [
    ModelSpec(
        name="TransMorphBayes",
        ckpt_subdir="experiments/TransMorphBayes_ncc_1_diffusion_1",
        backbone="TransMorphBayes",
        mc_iter=25,
    ),
    ModelSpec(
        name="HER_active",
        ckpt_subdir="her_transmorph/experiments/HER_her_active",
        backbone="TransMorph",
    ),
    ModelSpec(
        name="HER_fresh",
        ckpt_subdir="her_transmorph/experiments/HER_her_fresh",
        backbone="TransMorph",
    ),
    ModelSpec(
        name="HERGRAD_active",
        ckpt_subdir="her_transmorph_her_grad/experiments/HERGRAD_active_plus_grad",
        backbone="TransMorph",
    ),
    ModelSpec(
        name="HERGRAD_splither1e4",
        ckpt_subdir="her_transmorph_her_grad/experiments/HERGRAD_active_plus_grad_splither1e4",
        backbone="TransMorph",
    ),
    ModelSpec(
        name="HERGRAD_foldonly",
        ckpt_subdir="her_transmorph_her_grad/experiments/HERGRAD_foldonly_g20_wu20_her1_grad1",
        backbone="TransMorph",
    ),
    ModelSpec(
        name="HER_dsc0743",
        ckpt_subdir="experiments/TransMorph_IXI_HER_ncc_1.0_grad_1.0_her_1.0_a0_b0.02_g20",
        backbone="TransMorph",
        ckpt_file="dsc0.743.pth.tar",
    ),
]


def get_ckpt_dir(spec: ModelSpec) -> str:
    return os.path.join(TRANSMORPH_DIR, spec.ckpt_subdir)


def repo_root() -> str:
    return _REPO_ROOT
