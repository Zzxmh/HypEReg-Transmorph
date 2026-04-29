"""Hyperelastic regularization (HER) loss and diagnostics for 3D displacement fields.

Reference: Burger, Modersitzki, Ruthotto (2013), A Hyperelastic Regularization
Energy for Image Registration, SIAM J. Sci. Comput. See HER_Loss tech doc.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_func


def compute_jacobian_determinant_torch(u: torch.Tensor) -> torch.Tensor:
    """J = det(I + grad u) via forward differences.

    Args:
        u: displacement (B, 3, D, H, W) ordered (u_x, u_y, u_z).
    Returns:
        J: (B, D-1, H-1, W-1)
    """
    du_dx = u[:, :, 1:, :-1, :-1] - u[:, :, :-1, :-1, :-1]
    du_dy = u[:, :, :-1, 1:, :-1] - u[:, :, :-1, :-1, :-1]
    du_dz = u[:, :, :-1, :-1, 1:] - u[:, :, :-1, :-1, :-1]

    F11 = du_dx[:, 0] + 1.0
    F21 = du_dx[:, 1]
    F31 = du_dx[:, 2]

    F12 = du_dy[:, 0]
    F22 = du_dy[:, 1] + 1.0
    F32 = du_dy[:, 2]

    F13 = du_dz[:, 0]
    F23 = du_dz[:, 1]
    F33 = du_dz[:, 2] + 1.0

    J = (
        F11 * (F22 * F33 - F23 * F32)
        - F12 * (F21 * F33 - F23 * F31)
        + F13 * (F21 * F32 - F22 * F31)
    )
    return J


def length_loss(u: torch.Tensor) -> torch.Tensor:
    """mean(||F - I||_F^2) = mean(||grad u||_F^2)."""
    du_dx = u[:, :, 1:, :-1, :-1] - u[:, :, :-1, :-1, :-1]
    du_dy = u[:, :, :-1, 1:, :-1] - u[:, :, :-1, :-1, :-1]
    du_dz = u[:, :, :-1, :-1, 1:] - u[:, :, :-1, :-1, :-1]
    return (du_dx.pow(2) + du_dy.pow(2) + du_dz.pow(2)).mean()


def volume_loss(
    J: torch.Tensor, eps: float = 1e-3, penalty_type: str = "rational"
) -> torch.Tensor:
    Jc = torch.clamp(J, min=eps)
    if penalty_type == "rational":
        return ((J - 1.0).pow(2) / Jc).mean()
    if penalty_type == "symmetric":
        return (0.5 * (J - 1.0).pow(2) / Jc + 0.5 * (J - 1.0).pow(2)).mean()
    if penalty_type == "log_barrier":
        return (-torch.log(Jc + 1e-12)).mean()
    if penalty_type == "simple_quadratic":
        return (J - 1.0).pow(2).mean()
    raise ValueError(f"Unknown penalty_type: {penalty_type}")


def fold_loss(J: torch.Tensor, eps: float = 1e-3, power: int = 2) -> torch.Tensor:
    """mean(ReLU(eps - J)^p); zero gradient on healthy voxels (J >= eps)."""
    return F_func.relu(eps - J).pow(power).mean()


class HyperelasticLoss(nn.Module):
    """L_HER = alpha*length + beta*volume + gamma*fold."""

    VALID_ABLATIONS = (
        "full",
        "length_only",
        "volume_only",
        "fold_only",
        "length_volume",
    )

    def __init__(
        self,
        alpha_length: float = 0.05,
        beta_volume: float = 0.1,
        gamma_fold: float = 100.0,
        eps: float = 1e-3,
        fold_power: int = 2,
        penalty_type: str = "rational",
        ablation: str = "full",
    ):
        super().__init__()
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(
                f"ablation must be in {self.VALID_ABLATIONS}, got {ablation}"
            )
        self.alpha = float(alpha_length)
        self.beta = float(beta_volume)
        self.gamma = float(gamma_fold)
        self.eps = float(eps)
        self.fold_power = int(fold_power)
        self.penalty_type = penalty_type
        self.ablation = ablation
        self.last_components = {"length": 0.0, "volume": 0.0, "fold": 0.0}

    def forward(self, u: torch.Tensor, *_ignored) -> torch.Tensor:
        device = u.device
        zero = torch.zeros((), device=device)
        need_J = self.ablation in {"full", "volume_only", "fold_only", "length_volume"}
        J = compute_jacobian_determinant_torch(u) if need_J else None

        L_len = (
            length_loss(u)
            if self.ablation in {"full", "length_only", "length_volume"}
            else zero
        )
        L_vol = (
            volume_loss(J, self.eps, self.penalty_type)
            if self.ablation in {"full", "volume_only", "length_volume"}
            else zero
        )
        L_fold = (
            fold_loss(J, self.eps, self.fold_power)
            if self.ablation in {"full", "fold_only"}
            else zero
        )

        self.last_components = {
            "length": float(L_len.detach().item()) if isinstance(L_len, torch.Tensor) else 0.0,
            "volume": float(L_vol.detach().item()) if isinstance(L_vol, torch.Tensor) else 0.0,
            "fold": float(L_fold.detach().item()) if isinstance(L_fold, torch.Tensor) else 0.0,
        }
        return self.alpha * L_len + self.beta * L_vol + self.gamma * L_fold


def sdlog_jacobian(J: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """std(log J) for non-uniformity diagnosis (not for training)."""
    logJ = torch.log(torch.clamp(J, min=eps))
    return logJ.std()


def neg_jac_fraction(J: torch.Tensor) -> torch.Tensor:
    """Fraction of voxels with J <= 0 (non-diffeomorphic)."""
    return (J <= 0).float().mean()


def mean_abs_divergence_np(u_np: np.ndarray) -> float:
    """mean(|div u|); input (3, D, H, W)."""
    div = (
        np.gradient(u_np[0], axis=0)
        + np.gradient(u_np[1], axis=1)
        + np.gradient(u_np[2], axis=2)
    )
    return float(np.mean(np.abs(div)))
