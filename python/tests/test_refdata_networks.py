"""Compare the networks with the MATLAB reference data (refdata/networks2d, networks3d)."""

import re

import pytest
import torch

from _refdata import (
    array,
    case_files,
    cell,
    kwargs,
    load,
    names,
    params_dict,
    text,
    tolerance,
    torch_dtype,
)
from saivdr.dcnn import create_nsolt_2d, create_nsolt_3d
from saivdr.dcnn.matlab_compat import load_params, rotation_layers, set_params

FILES = case_files("networks2d") + case_files("networks3d")


def build(m, ndims):
    kw = kwargs(text(m, "meta_params"))
    create = create_nsolt_2d if ndims == 2 else create_nsolt_3d
    return create(dtype=torch.float64, **kw)


def positions(port_names, nlevels):
    """Map the MATLAB input/output names to the Python positions."""
    out = []
    for name in port_names:
        mm = re.search(r"Lv(\d+)_(Ac|Dc)", name)
        if mm is None:
            out.append(0)
        else:
            out.append(int(mm.group(1)) - 1 if mm.group(2) == "Ac" else nlevels)
    return out


def as_tuple(z):
    return z if isinstance(z, tuple) else (z,)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_network(path):
    m = load(path)
    ndims = int(m["meta_ndims"].item())
    dtype = text(m, "meta_dtype")
    tdtype = torch_dtype(dtype)
    kind = f"image{ndims}d"
    net = build(m, ndims)
    set_params(net, params_dict(m))
    nlevels = int(kwargs(text(m, "meta_params"))["number_of_levels"])
    in_names = text(m, "meta_input_names").split(",")
    out_names = text(m, "meta_output_names").split(",")

    X = [None] * len(in_names)
    for i, pos in enumerate(positions(in_names, nlevels)):
        X[pos] = array(m, f"x{i + 1}", kind, tdtype).requires_grad_(True)
    Z = as_tuple(net(*X))
    assert len(Z) == len(out_names)

    rtol, atol = tolerance(dtype)
    loss = 0
    for i, pos in enumerate(positions(out_names, nlevels)):
        torch.testing.assert_close(Z[pos], array(m, f"z{i + 1}", kind, tdtype), rtol=rtol, atol=atol)
        loss = loss + (Z[pos] * array(m, f"dldz{i + 1}", kind, tdtype)).sum()
    loss.backward()

    rtol, atol = tolerance(dtype, gradient=True)
    for i, pos in enumerate(positions(in_names, nlevels)):
        torch.testing.assert_close(X[pos].grad, array(m, f"dldx{i + 1}", kind, tdtype), rtol=rtol, atol=atol)
    rot = rotation_layers(net)
    for name, dldw in zip(names(m), cell(m, "dldw")):
        expected = torch.as_tensor(dldw.reshape(-1), dtype=torch.float64)
        torch.testing.assert_close(rot[name].angles.grad, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("path", case_files("params"), ids=lambda p: p.stem)
def test_load_params(path):
    """Parameters exported by saivdr_export_params give the same outputs."""
    m = load(path)
    net = build(m, int(m["meta_ndims"].item()))
    load_params(net, path)
    nlevels = int(kwargs(text(m, "meta_params"))["number_of_levels"])
    Z = as_tuple(net(array(m, "x1", "image2d")))
    for i, pos in enumerate(positions(text(m, "meta_output_names").split(","), nlevels)):
        torch.testing.assert_close(Z[pos], array(m, f"z{i + 1}", "image2d"), rtol=1e-9, atol=1e-10)
