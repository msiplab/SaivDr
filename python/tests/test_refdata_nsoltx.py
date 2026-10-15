"""nsoltx filter banks exported by saivdr_export_nsoltx agree with generalfb (refdata/nsoltx)."""

import pytest
import torch

from _refdata import case_files, load
from saivdr.dcnn import NsoltSubbandDeserialization2dLayer, NsoltSubbandSerialization2dLayer
from saivdr.dcnn.matlab_compat import from_matlab_array, load_nsoltx

FILES = case_files("nsoltx")


def serialization_layers(net, size):
    kw = dict(
        original_dimension=size,
        number_of_channels=net.number_of_channels,
        decimation_factor=net.decimation_factor,
        number_of_levels=net.number_of_levels,
    )
    return NsoltSubbandSerialization2dLayer(**kw), NsoltSubbandDeserialization2dLayer(**kw)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_analysis_and_synthesis(path):
    m = load(path)
    net = load_nsoltx(path, dtype=torch.float64)
    x = from_matlab_array(m["x"], "image2d", dtype=torch.float64)
    srz, dsz = serialization_layers(net.analyzer, tuple(x.shape[-2:]))
    coefs = torch.as_tensor(m["coefs"].ravel(), dtype=torch.float64)
    torch.testing.assert_close(srz(*net.analyzer(x))[0], coefs, rtol=1e-9, atol=1e-10)
    y = torch.as_tensor(m["y"].ravel(), dtype=torch.float64).unsqueeze(0)
    expected = from_matlab_array(m["synthesized"], "image2d", dtype=torch.float64)
    torch.testing.assert_close(net.synthesizer(*dsz(y)), expected, rtol=1e-9, atol=1e-10)


def test_analyzer_only(tmp_path):
    path = FILES[0]
    ana = load_nsoltx(path, mode="Analyzer", dtype=torch.float64)
    whole = load_nsoltx(path, dtype=torch.float64)
    x = torch.rand(1, 1, 32, 32, dtype=torch.float64)
    for a, b in zip(ana(x), whole.analyzer(x)):
        torch.testing.assert_close(a, b)


def test_number_of_levels_override():
    """A 1-level export with number_of_levels=4 equals the 4-level export."""
    lv1 = [p for p in FILES if p.stem == "Nsoltx_designed_lena128_lv1"][0]
    lv4 = [p for p in FILES if p.stem == "Nsoltx_designed_lena128_lv4"][0]
    a = load_nsoltx(lv1, mode="Analyzer", number_of_levels=4, dtype=torch.float64)
    b = load_nsoltx(lv4, mode="Analyzer", dtype=torch.float64)
    x = torch.rand(1, 1, 64, 64, dtype=torch.float64)
    for u, v in zip(a(x), b(x)):
        torch.testing.assert_close(u, v)
