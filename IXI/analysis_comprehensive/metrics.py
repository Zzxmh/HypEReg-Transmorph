"""
Overlap (A), surface (B), intensity (C), and deformation (D) metrics.
Per-VOI: 30 IXI VOIs. NaN if structure missing in both pred and gt? — use union empty → nan.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import math

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

# Default 1mm isotropic voxels in physical space (IXI)
DEFAULT_SPACING = (1.0, 1.0, 1.0)
_STRUCT3 = np.ones((3, 3, 3), dtype=bool)


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m
    er = binary_erosion(m, structure=_STRUCT3, border_value=0)
    return m ^ er


def dice_jaccard_vs(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float, float]:
    """Single binary mask. Returns (dice, jaccard, volume_similarity)."""
    p = pred.astype(bool)
    t = true.astype(bool)
    inter = np.logical_and(p, t).sum()
    s_p, s_t = p.sum(), t.sum()
    if s_p == 0 and s_t == 0:
        return float("nan"), float("nan"), float("nan")
    union = s_p + s_t
    union_or = s_p + s_t - inter
    dice = (2.0 * inter) / (union + 1e-8)
    jacc = float(inter) / (union_or + 1e-8)
    vs = 1.0 - abs(float(s_p) - float(s_t)) / (float(union) + 1e-8)
    return float(dice), float(jacc), float(vs)


def overlap_per_voi(
    def_pred: np.ndarray, y_true: np.ndarray, voi_lbs: List[int]
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    pred = np.asarray(def_pred, dtype=np.int32)
    true = np.asarray(y_true, dtype=np.int32)
    dice: Dict[int, float] = {}
    jacc: Dict[int, float] = {}
    vss: Dict[int, float] = {}
    for lb in voi_lbs:
        p_m = pred == lb
        t_m = true == lb
        d, j, v = dice_jaccard_vs(p_m, t_m)
        dice[lb] = d
        jacc[lb] = j
        vss[lb] = v
    return dice, jacc, vss


def hd95_assd_nsd(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    tau: float = 1.0,
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
) -> Tuple[float, float, float]:
    """Returns (hd95_mm, assd_mm, surface_dice_nsd) or (nan, nan, nan) if no surface."""
    p = pred_mask.astype(bool)
    t = true_mask.astype(bool)
    if not p.any() and not t.any():
        return float("nan"), float("nan"), float("nan")
    if not p.any() or not t.any():
        return float("nan"), float("nan"), float("nan")

    sp, sg = _surface_voxels(p), _surface_voxels(t)
    if not sp.any() or not sg.any():
        return float("nan"), float("nan"), float("nan")

    dt_t = distance_transform_edt(~t, sampling=spacing)
    dt_p = distance_transform_edt(~p, sampling=spacing)
    d_p2t = dt_t[sp]
    d_t2p = dt_p[sg]
    hd = float(
        max(np.percentile(d_p2t, 95), np.percentile(d_t2p, 95))
    )
    assd = float(
        (d_p2t.sum() + d_t2p.sum()) / (sp.sum() + sg.sum())
    )
    match = (d_p2t <= tau).sum() + (d_t2p <= tau).sum()
    nsd = float(match / (sp.sum() + sg.sum()))
    return hd, assd, nsd


def surface_per_voi(
    def_pred: np.ndarray,
    y_true: np.ndarray,
    voi_lbs: List[int],
    tau: float = 1.0,
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    pred = np.asarray(def_pred, dtype=np.int32)
    true = np.asarray(y_true, dtype=np.int32)
    out_hd: Dict[int, float] = {}
    out_as: Dict[int, float] = {}
    out_ns: Dict[int, float] = {}
    for lb in voi_lbs:
        h, a, n = hd95_assd_nsd(pred == lb, true == lb, tau=tau, spacing=spacing)
        out_hd[lb] = h
        out_as[lb] = a
        out_ns[lb] = n
    return out_hd, out_as, out_ns


def ncc_global_numpy(warped: np.ndarray, fixed: np.ndarray) -> float:
    a = np.asarray(warped, dtype=np.float64).ravel()
    b = np.asarray(fixed, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8)
    return float((a * b).sum() / den)


def mse_psnr(warped: np.ndarray, fixed: np.ndarray, data_range: float = 1.0) -> Tuple[float, float]:
    a = np.asarray(warped, dtype=np.float64)
    b = np.asarray(fixed, dtype=np.float64)
    m = float(np.mean((a - b) ** 2))
    if m <= 0:
        psnr = 99.0
    else:
        psnr = float(10.0 * np.log10((data_range ** 2) / (m + 1e-12)))
    return m, psnr


def mutual_info_and_nmi(
    warped: np.ndarray, fixed: np.ndarray, bins: int = 64
) -> Tuple[float, float]:
    x = np.asarray(warped, dtype=np.float64).ravel()
    y = np.asarray(fixed, dtype=np.float64).ravel()
    h2d, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = h2d.astype(np.float64)
    s = pxy.sum()
    if s <= 0:
        return 0.0, 1.0
    pxy /= s
    px1 = pxy.sum(axis=1)
    py1 = pxy.sum(axis=0)
    px_py = np.outer(px1, py1)
    eps = 1e-12
    nz = pxy > 0
    mi = float(
        (pxy[nz] * (np.log(pxy[nz] + eps) - np.log(px_py[nz] + eps))).sum()
    )
    hx = float(-(px1[px1 > 0] * np.log(px1[px1 > 0] + eps)).sum())
    hy = float(-(py1[py1 > 0] * np.log(py1[py1 > 0] + eps)).sum())
    hxy = float(-(pxy[pxy > 0] * np.log(pxy[pxy > 0] + eps)).sum())
    nmi = float((hx + hy) / (hxy + eps)) if hxy > eps else 2.0
    return mi, nmi


def jacobian_field_stats(
    J: np.ndarray, eps: float = 1e-6
) -> Dict[str, float]:
    J = np.asarray(J, dtype=np.float64)
    flat = J.ravel()
    neg = float((flat <= 0).mean())
    j_pos = J[J > 0]
    if j_pos.size == 0:
        sd = float("nan")
    else:
        sd = float(np.log(j_pos.clip(eps, None)).std())
    return {
        "non_jac_frac": float(neg),
        "SDlogJ": sd,
        "J_min": float(J.min()),
        "J_p01": float(np.percentile(flat, 1)),
        "J_p05": float(np.percentile(flat, 5)),
        "J_p50": float(np.percentile(flat, 50)),
        "J_p95": float(np.percentile(flat, 95)),
        "J_p99": float(np.percentile(flat, 99)),
        "J_max": float(J.max()),
    }


def mean_abs_divergence_np(flow_3dhw: np.ndarray) -> float:
    u = flow_3dhw
    if u.shape[0] != 3:
        raise ValueError("flow must be (3, D, H, W)")
    du_dx = (u[0, 2:, 1:-1, 1:-1] - u[0, :-2, 1:-1, 1:-1]) * 0.5
    du_dy = (u[1, 1:-1, 2:, 1:-1] - u[1, 1:-1, :-2, 1:-1]) * 0.5
    du_dz = (u[2, 1:-1, 1:-1, 2:] - u[2, 1:-1, 1:-1, :-2]) * 0.5
    div = du_dx + du_dy + du_dz
    return float(np.mean(np.abs(div)))


def bending_energy_full(flow: np.ndarray, spacing: Tuple[float, float, float] = DEFAULT_SPACING) -> float:
    """
    Voxel-averaged sum of squared second partials of u_i w.r.t. all spatial pairs (plan §D.5).
    flow: (3, D, H, W)
    """
    u = flow.astype(np.float64)
    acc = 0.0
    n = 0
    for c in range(3):
        g0 = np.gradient(u[c], spacing[0], spacing[1], spacing[2])
        for d in range(3):
            h = np.gradient(g0[d], spacing[0], spacing[1], spacing[2])
            for e in range(3):
                acc += float(np.sum(h[e] * h[e]))
                n += h[e].size
    if n == 0:
        return 0.0
    return acc / n


def subsample_log_j_pos(J: np.ndarray, max_voxels: int = 200_000, seed: int = 0) -> np.ndarray:
    J = np.asarray(J, dtype=np.float64)
    pos = J[J > 0]
    if pos.size == 0:
        return np.array([], dtype=np.float64)
    rng = np.random.default_rng(seed)
    if pos.size > max_voxels:
        idx = rng.choice(pos.size, size=max_voxels, replace=False)
        pos = pos[idx]
    return np.log(np.clip(pos, 1e-8, None))
