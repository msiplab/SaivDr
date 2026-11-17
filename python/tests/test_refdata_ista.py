"""ISTA restoration agrees with IstaImRestoration2d in MATLAB (refdata/ista)."""

import pytest
import torch

import saivdr.degradation as dg
from _refdata import array, case_files, load, text
from saivdr.dcnn.matlab_compat import load_nsoltx
from saivdr.restoration import IstaImRestoration2d

FILES = case_files("ista")


def linear_process(m):
    kind = text(m, "meta_degradation")
    if kind == "db":
        return dg.BlurSystem(blur_type="Gaussian", sigma_of_gaussian_kernel=2)
    if kind == "sr":
        return dg.DecimationSystem(
            2, 2, blur_type="Gaussian", sigma_of_gaussian_kernel=2, tol_of_power_method=1e-6
        )
    return dg.PixelLossSystem(loss_type="Specified", mask=torch.as_tensor(m["mask"]))


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
@pytest.mark.parametrize("estimate_lambda_max", [False, True])
def test_ista(path, estimate_lambda_max):
    m = load(path)
    net = load_nsoltx(path, dtype=torch.float64)
    lin = linear_process(m)
    # MATLAB estimates LambdaMax when the degradation is applied to the
    # original image, before the restoration; for DecimationSystem this is
    # the value for an image twice as large as the original
    lmax = lin.lambda_max(tuple(array(m, "original", "image2d").shape[-2:])) if estimate_lambda_max else None
    ista = IstaImRestoration2d(
        synthesizer=net.synthesizer,
        adj_of_synthesizer=net.analyzer,
        linear_process=lin,
        lambda_=float(m["lambda"].item()),
        max_iter=int(m["max_iter"].item()),
        lambda_max=lmax if estimate_lambda_max else float(m["lambda_max"].item()),
    )
    restored = ista(array(m, "observed", "image2d"))
    # With the power-method estimate, lambda_max agrees with MATLAB up to its tolerance
    tol = 1e-5 if estimate_lambda_max else 1e-9
    torch.testing.assert_close(restored, array(m, "restored", "image2d"), rtol=tol, atol=tol)
