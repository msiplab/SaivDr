"""Linear processes agree with saivdr.degradation in MATLAB (refdata/degradation)."""

import json

import pytest
import torch

import saivdr.degradation as dg
from _refdata import array, case_files, load, text

FILES = case_files("degradation")

_KWARGS = {
    "BlurType": "blur_type",
    "SigmaOfGaussianKernel": "sigma_of_gaussian_kernel",
    "SizeOfKernel": "size_of_kernel",
    "BoundaryOption": "boundary_option",
    "CustomKernel": "custom_kernel",
    "VerticalDecimationFactor": "vertical_decimation_factor",
    "HorizontalDecimationFactor": "horizontal_decimation_factor",
    "LossType": "loss_type",
    "Density": "density",
    "Seed": "seed",
}


def make(m):
    params = {_KWARGS[k]: v for k, v in json.loads(text(m, "meta_params")).items()}
    cls = text(m, "meta_class")
    if cls == "PixelLossSystem":
        # The MATLAB random mask cannot be reproduced; use it as a specified mask
        return dg.PixelLossSystem(loss_type="Specified", mask=torch.as_tensor(m["mask"]))
    return getattr(dg, cls)(**params)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_linear_process(path):
    m = load(path)
    proc = make(m)
    x = array(m, "x", "image2d")
    torch.testing.assert_close(proc.forward(x), array(m, "z", "image2d"), rtol=1e-12, atol=1e-13)
    y = array(m, "y", "image2d")
    torch.testing.assert_close(proc.adjoint(y), array(m, "adjoint", "image2d"), rtol=1e-12, atol=1e-13)
    if all(s % 2 for s in getattr(proc, "blur_kernel", torch.ones(1, 1)).shape):
        # Adjointness <P x, y> = <x, P^T y>, which MATLAB keeps only for odd kernel sizes
        torch.testing.assert_close((proc.forward(x) * y).sum(), (x * proc.adjoint(y)).sum())
    # MATLAB estimates LambdaMax when the process is first called, here with x
    lmax = proc.lambda_max(tuple(x.shape[-2:]))
    assert lmax == pytest.approx(float(m["lambda_max"].item()), rel=1e-6)
