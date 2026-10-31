import numpy as np
import pytest
import torch

from saivdr.dcnn import create_nsolt_2d
from saivdr.dcnn.matlab_compat import (
    from_matlab_array,
    get_params,
    load_params,
    save_params,
    set_params,
    to_matlab_array,
)


@pytest.mark.parametrize(
    "kind, shape, index",
    [
        # MATLAB element (h, w, c, n) -> PyTorch element (n, c, h, w)
        ("image2d", (4, 5, 3, 2), lambda h, w, c, n: (n, c, h, w)),
        ("coef2d", (3, 4, 5, 2), lambda c, r, k, n: (n, c, r, k)),
        ("image3d", (4, 5, 6, 3, 2), lambda h, w, d, c, n: (n, c, d, h, w)),
        ("coef3d", (3, 4, 5, 6, 2), lambda c, r, k, lay, n: (n, c, lay, r, k)),
    ],
)
def test_array_layout(kind, shape, index):
    a = np.random.default_rng(0).standard_normal(shape)
    t = from_matlab_array(a, kind)
    for _ in range(10):
        i = tuple(np.random.randint(s) for s in shape)
        assert t[index(*i)].item() == a[i]
    np.testing.assert_array_equal(to_matlab_array(t, kind), a)


def test_trailing_singletons_and_serial():
    a = np.arange(12.0).reshape(4, 3, order="F")  # MATLAB 4 x 3 x 1 x 1
    assert from_matlab_array(a, "image2d").shape == (1, 1, 4, 3)
    assert from_matlab_array(a, "image2d", size=[4, 3, 1, 1]).shape == (1, 1, 4, 3)
    s = np.arange(10.0).reshape(5, 1, 1, 2, order="F")  # nElements x 1 x 1 x N
    t = from_matlab_array(s, "serial")
    np.testing.assert_array_equal(t.numpy(), np.arange(10.0).reshape(2, 5))
    np.testing.assert_array_equal(to_matlab_array(t, "serial"), s)


def test_params_round_trip(tmp_path):
    src = create_nsolt_2d(
        number_of_channels=(3, 3),
        polyphase_order=(2, 2),
        number_of_levels=2,
        number_of_vanishing_moments=(1, 0),
        dtype=torch.float64,
    )
    with torch.no_grad():
        for p in src.parameters():
            p.normal_()
    path = tmp_path / "params.mat"
    save_params(src, path)
    dst = create_nsolt_2d(
        number_of_channels=(3, 3),
        polyphase_order=(2, 2),
        number_of_levels=2,
        number_of_vanishing_moments=(0, 1),
        dtype=torch.float64,
    )
    load_params(dst, path)
    a, b = get_params(src), get_params(dst)
    assert a.keys() == b.keys()
    for name in a:
        np.testing.assert_array_equal(a[name]["Angles"], b[name]["Angles"])
        np.testing.assert_array_equal(a[name]["Mus"], b[name]["Mus"])
        assert a[name].get("NoDcLeakage") == b[name].get("NoDcLeakage")
    x = torch.randn(1, 1, 8, 8, dtype=torch.float64)
    torch.testing.assert_close(dst(x), src(x))


def test_set_params_strict():
    net = create_nsolt_2d(mode="Analyzer")
    with pytest.raises(KeyError):
        set_params(net, {"Lv9_Cmp1_V0": {"Angles": [0.0], "Mus": [1, 1, 1, 1]}})
