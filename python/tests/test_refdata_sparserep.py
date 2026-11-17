"""Iterative hard thresholding agrees with MATLAB (refdata/sparserep)."""

import pytest
import torch

from _refdata import array, case_files, load
from saivdr.dcnn import NsoltSubbandSerialization2dLayer
from saivdr.dcnn.matlab_compat import load_nsoltx
from saivdr.sparserep import IterativeHardThresholding

FILES = case_files("sparserep")


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_iht(path):
    m = load(path)
    net = load_nsoltx(path, dtype=torch.float64)
    iht = IterativeHardThresholding(
        net.synthesizer,
        net.analyzer,
        number_of_sparse_coefficients=int(m["n_coefs"].item()),
        max_iter=int(m["max_iter"].item()),
    )
    source = array(m, "source", "image2d")
    result, coefs = iht(source)
    torch.testing.assert_close(result, array(m, "result", "image2d"), rtol=1e-9, atol=1e-10)
    srz = NsoltSubbandSerialization2dLayer(
        original_dimension=tuple(source.shape[-2:]),
        number_of_channels=net.analyzer.number_of_channels,
        decimation_factor=net.analyzer.decimation_factor,
        number_of_levels=net.analyzer.number_of_levels,
    )
    expected = torch.as_tensor(m["coefs"].ravel(), dtype=torch.float64)
    torch.testing.assert_close(srz(*coefs)[0], expected, rtol=1e-9, atol=1e-10)
