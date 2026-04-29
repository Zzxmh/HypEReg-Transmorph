# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
import torch

from ._helpers import cfg_with_size

PREFERRED_DEVICE = "cpu"


class _ClassicalModel:
    def to(self, _device):
        return self

    def eval(self):
        return self


def build_model(device: str = "cuda"):
    _ = device
    return _ClassicalModel(), cfg_with_size((160, 192, 224))


def _run_affine(moving_np: np.ndarray, fixed_np: np.ndarray):
    moving_s = sitk.GetImageFromArray(moving_np.astype(np.float32))
    fixed_s = sitk.GetImageFromArray(fixed_np.astype(np.float32))
    moving_s.SetSpacing((1.0, 1.0, 1.0))
    fixed_s.SetSpacing((1.0, 1.0, 1.0))

    tx0 = sitk.CenteredTransformInitializer(
        fixed_s,
        moving_s,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=120, convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(tx0, inPlace=False)
    tx = reg.Execute(fixed_s, moving_s)

    warped = sitk.Resample(
        moving_s, fixed_s, tx, sitk.sitkLinear, 0.0, moving_s.GetPixelID()
    )
    disp = sitk.TransformToDisplacementField(
        tx,
        sitk.sitkVectorFloat64,
        fixed_s.GetSize(),
        fixed_s.GetOrigin(),
        fixed_s.GetSpacing(),
        fixed_s.GetDirection(),
    )
    warped_np = sitk.GetArrayFromImage(warped).astype(np.float32)
    disp_np = sitk.GetArrayFromImage(disp).astype(np.float32)  # (D,H,W,3)
    flow = np.moveaxis(disp_np, -1, 0)
    return warped_np, flow


def forward(model, x, y):
    _ = model
    moving = x.detach().cpu().numpy()[0, 0].astype(np.float32)
    fixed = y.detach().cpu().numpy()[0, 0].astype(np.float32)
    warped_np, flow = _run_affine(moving, fixed)
    warped_t = torch.from_numpy(warped_np[None, None]).to(x.device)
    flow_t = torch.from_numpy(flow[None]).to(x.device)
    return warped_t, flow_t
