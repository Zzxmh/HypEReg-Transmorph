# -*- coding: utf-8 -*-
"""
Full IXI evaluation metrics (numpy / scipy / skimage).
Jacobian uses the same jacobian_determinant_vxm as IXI/TransMorph/utils.py.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from scipy import ndimage

# Optional: skimage for SSIM
try:
    from skimage.metrics import structural_similarity as sk_ssim
except Exception:  # pragma: no cover
    sk_ssim = None  # type: ignore


def _ensure_transmorph_path():
    ixi_dir = os.path.dirname(os.path.abspath(__file__))
    tm = os.path.join(ixi_dir, "TransMorph")
    if tm not in sys.path:
        sys.path.insert(0, tm)


def jacobian_determinant_vxm_np(disp_czyx: NDArray) -> NDArray:
    """
    Displacement: shape (3, D, H, W) in torch channel order, same as flow.cpu().numpy()[0].
    """
    _ensure_transmorph_path()
    import utils as tm_utils  # noqa: WPS433

    return tm_utils.jacobian_determinant_vxm(disp_czyx)


def per_label_dice(
    y_pred: NDArray, y_true: NDArray, num_classes: int = 46
) -> Tuple[NDArray, NDArray]:
    """
    Label maps: integer (D,H,W) with values 0..num_classes-1.
    Returns: (dice_per_class (num_classes,), valid_mask) — invalid => nan
    """
    dices = np.full(num_classes, np.nan, dtype=np.float64)
    valid = np.zeros(num_classes, dtype=bool)
    for c in range(num_classes):
        p = y_pred == c
        t = y_true == c
        inter = (p & t).sum()
        union = p.sum() + t.sum()
        # Match utils.dice_val_substruct: both empty -> 0 / 1e-5 = 0
        dices[c] = (2.0 * inter) / (union + 1e-5)
        valid[c] = True
    return dices, valid


def jaccard_from_masks(pred: NDArray, true: NDArray) -> float:
    p = pred.astype(bool)
    t = true.astype(bool)
    if not t.any() and not p.any():
        return float("nan")
    inter = (p & t).sum()
    union = p | t
    if not union.any():
        return float("nan")
    return inter / (union.sum() + 1e-8)


def jaccard_mean_over_labels(
    y_pred: NDArray, y_true: NDArray, num_classes: int = 46
) -> float:
    vals = []
    for c in range(num_classes):
        p = y_pred == c
        t = y_true == c
        if not t.any() and not p.any():
            continue
        v = jaccard_from_masks(p, t)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def volume_similarity(
    y_pred: NDArray, y_true: NDArray, num_classes: int = 46
) -> NDArray:
    """
    Per-class VS = 1 - ||S|-|G|| / (|S|+|G|) (range [-1,1]).
    """
    out = np.full(num_classes, np.nan, dtype=np.float64)
    for c in range(num_classes):
        p = (y_pred == c).sum()
        t = (y_true == c).sum()
        if p + t == 0:
            out[c] = 1.0
            continue
        out[c] = 1.0 - abs(p - t) / (p + t + 1e-8)
    return out


def _surface_voxels(m: NDArray) -> NDArray:
    m = m.astype(bool)
    if not m.any():
        return np.zeros_like(m, dtype=bool)
    ero = ndi.binary_erosion(m, iterations=1)
    return m & (~ero)


def _hausdorff_d95_cdist(a_surf, b_surf, spacing) -> float:
    """Asymmetric: 95th percentile of min distances from a_surf to b."""
    if not a_surf.any() or not b_surf.any():
        return float("nan")
    dt_b = ndi.distance_transform_edt(~b_surf, sampling=spacing)
    d_on_a = dt_b[a_surf]
    d_on_a = d_on_a[np.isfinite(d_on_a)]
    if d_on_a.size == 0:
        return float("nan")
    return float(np.quantile(d_on_a, 0.95))


def hd95_one_structure(
    pred: NDArray, true: NDArray, spacing: Union[float, Sequence[float]] = 1.0
) -> float:
    p = pred.astype(bool)
    t = true.astype(bool)
    if p.sum() == 0 or t.sum() == 0:
        return float("nan")
    s1 = _surface_voxels(p)
    s2 = _surface_voxels(t)
    if not s1.any() and not s2.any():
        return 0.0
    if not s1.any() or not s2.any():
        return float("nan")
    d12 = _hausdorff_d95_cdist(s1, s2, spacing)
    d21 = _hausdorff_d95_cdist(s2, s1, spacing)
    return max(d12, d21)


def assd_one_structure(
    pred: NDArray, true: NDArray, spacing: Union[float, Sequence[float]] = 1.0
) -> float:
    p = pred.astype(bool)
    t = true.astype(bool)
    if p.sum() == 0 or t.sum() == 0:
        return float("nan")
    s1 = _surface_voxels(p)
    s2 = _surface_voxels(t)
    if not s1.any() and not s2.any():
        return 0.0
    if not s1.any() or not s2.any():
        return float("nan")
    dt1 = ndi.distance_transform_edt(~s1, sampling=spacing)
    dt2 = ndi.distance_transform_edt(~s2, sampling=spacing)
    d1 = dt2[s1].mean() if s1.any() else np.nan
    d2 = dt1[s2].mean() if s2.any() else np.nan
    if not np.isfinite(d1) or not np.isfinite(d2):
        return float("nan")
    return float(0.5 * (d1 + d2))


def hd95_assd_mean_over_labels(
    y_pred: NDArray,
    y_true: NDArray,
    num_classes: int = 46,
    spacing: Union[float, Sequence[float]] = 1.0,
) -> Tuple[float, float]:
    hds, ass = [], []
    for c in range(num_classes):
        p = y_pred == c
        t = y_true == c
        if p.sum() == 0 and t.sum() == 0:
            continue
        h = hd95_one_structure(p, t, spacing)
        a = assd_one_structure(p, t, spacing)
        if np.isfinite(h):
            hds.append(h)
        if np.isfinite(a):
            ass.append(a)
    return (
        float(np.nanmean(hds)) if hds else float("nan"),
        float(np.nanmean(ass)) if ass else float("nan"),
    )


def ssim3d(
    a: NDArray,
    b: NDArray,
    data_range: float = 1.0,
    win_size: int = 7,
) -> float:
    """
    Mean 2D SSIM over axial (z) slices, using skimage.
    """
    if sk_ssim is None:
        return float("nan")
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("nan")
    w = min(int(win_size), a.shape[0], a.shape[1], a.shape[2])
    if w % 2 == 0:
        w = max(3, w - 1)
    if w < 3:
        return float("nan")
    ss = []
    for z in range(a.shape[0]):
        s = sk_ssim(
            a[z],
            b[z],
            data_range=data_range,
            win_size=min(w, a.shape[1], a.shape[2]),
        )
        ss.append(s)
    return float(np.mean(ss))


def nmi_3d(
    fixed: NDArray,
    moving: NDArray,
    bins: int = 64,
    eps: float = 1e-12,
) -> float:
    """
    Normalized mutual information (histogram-based, marginal equalization style).
    """
    a = np.asarray(fixed, dtype=np.float64).ravel()
    b = np.asarray(moving, dtype=np.float64).ravel()
    a = np.clip(a, 0, 1)
    b = np.clip(b, 0, 1)
    h2d, _, _ = np.histogram2d(
        a, b, bins=bins, range=[[0, 1 + eps], [0, 1 + eps]]
    )
    pxy = h2d / (h2d.sum() + eps)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    pa = pxy * np.log2((pxy + eps) / (px @ py + eps))
    mi = float(np.sum(pa[np.isfinite(pa)]))
    hx = -np.sum(px * np.log2(px + eps))
    hy = -np.sum(py * np.log2(py + eps))
    if hx <= 0 or hy <= 0:
        return float("nan")
    return 2.0 * mi / (hx + hy + eps)


def jacobian_stats(
    flow_czyx: NDArray,
    hist_range: Optional[Tuple[float, float]] = None,
    n_bins: int = 100,
) -> dict:
    """
    flow: (3, D, H, W). Returns non_jec, SDlogJ, percentiles, min/max, hist counts.
    non_jec = fraction of voxels with J <= 0 (same as analysis_trans).
    """
    J = jacobian_determinant_vxm_np(flow_czyx).astype(np.float64)
    flat = J.ravel()
    n = flat.size
    non_jec = float(np.sum(flat <= 0) / (n + 1e-12))
    pos = flat[flat > 0]
    if pos.size == 0:
        sdlog = float("nan")
    else:
        sdlog = float(np.std(np.log(pos + 1e-8)))
    p01, p50, p99 = (
        float(np.quantile(flat, 0.01)),
        float(np.quantile(flat, 0.50)),
        float(np.quantile(flat, 0.99)),
    )
    j_min, j_max = float(flat.min()), float(flat.max())
    if hist_range is None:
        lo, hi = j_min, j_max
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
    else:
        lo, hi = hist_range
    h, e = np.histogram(flat, bins=n_bins, range=(lo, hi))
    return {
        "non_jec": non_jec,
        "SDlogJ": sdlog,
        "J_p01": p01,
        "J_p50": p50,
        "J_p99": p99,
        "J_min": j_min,
        "J_max": j_max,
        "J_flat": J,
        "hist_counts": h.astype(int).tolist(),
        "hist_edges": e.astype(float).tolist(),
    }


def _warp3d(
    field_dhw3: NDArray, disp_fwd_czyx: NDArray, order: int = 1
) -> NDArray:
    """
    Tri-linear warp: sample field at (idx + disp). field shape (D,H,W,3), disp (3,D,H,W).
    """
    D, H, W = field_dhw3.shape[:3]
    z, y, x = np.mgrid[0:D, 0:H, 0:W].astype(np.float32)
    dz, dy, dx = [disp_fwd_czyx[i] for i in range(3)]
    coords = np.stack(
        (z + dz, y + dy, x + dx), axis=0
    )  # (3, D, H, W)
    out = np.empty((D, H, W, 3), dtype=np.float64)
    for k in range(3):
        out[..., k] = ndimage.map_coordinates(
            field_dhw3[..., k], coords, order=order, mode="reflect"
        )
    return out


def inverse_consistency_error(
    phi_fwd: NDArray,
    phi_bwd: NDArray,
    spacing: Optional[Sequence[float]] = None,
) -> float:
    """
    Mean || (id + u_fwd) + u_bwd ∘ (id + u_fwd) - id || in voxel units, simplified ICE.
    phi_fwd, phi_bwd: (3, D, H, W) displacements in index space.
    """
    d = phi_fwd[0]
    D, H, W = d.shape
    # backward field in moving space -> warp by forward composition
    bwd = np.stack([phi_bwd[0], phi_bwd[1], phi_bwd[2]], axis=-1)
    warped_bwd = _warp3d(bwd, phi_fwd)
    comp = warped_bwd + np.stack([phi_fwd[0], phi_fwd[1], phi_fwd[2]], axis=-1)
    res = comp  # (x + u_fwd) o ... actually id + u_fwd + u_bwd( x+u_fwd) should be x
    err = res  # we want (u_fwd + warp(u_bwd))  ~ 0
    mag = np.sqrt(np.sum(err ** 2, axis=-1))
    if spacing is not None:
        s = np.asarray(spacing, dtype=np.float64).reshape(1, 1, 1, 3)
        mag = mag * float(np.sqrt(np.mean(s ** 2)))
    return float(mag.mean())
