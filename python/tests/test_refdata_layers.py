"""Compare each layer with the MATLAB reference data (refdata/layers)."""

import pytest
import torch

import saivdr.dcnn.layers as layers
from _refdata import array, case_files, kwargs, load, text, tolerance, torch_dtype

FILES = case_files("layers")


def make_layer(m):
    name = text(m, "meta_layer")
    cls = getattr(layers, "N" + name[1:])
    kw = kwargs(text(m, "meta_params"))
    if "angles" in m:
        kw["dtype"] = torch.float64
    layer = cls(**kw)
    if "angles" in m:
        # Raw values as stored in the MATLAB layer
        layer.set_angles(m["angles"].ravel())
        layer.set_mus(m["mus"].ravel())
    return layer


def as_tuple(z):
    return z if isinstance(z, tuple) else (z,)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_layer(path):
    m = load(path)
    dtype = text(m, "meta_dtype")
    tdtype = torch_dtype(dtype)
    in_kinds = text(m, "meta_in_kinds").split(",")
    out_kinds = text(m, "meta_out_kinds").split(",")
    backward = bool(m["meta_backward"].item())
    X = [array(m, f"x{i + 1}", k, tdtype).requires_grad_(backward) for i, k in enumerate(in_kinds)]
    layer = make_layer(m)

    Z = as_tuple(layer(*X))
    assert len(Z) == len(out_kinds)
    rtol, atol = tolerance(dtype)
    for i, (z, kind) in enumerate(zip(Z, out_kinds)):
        torch.testing.assert_close(z, array(m, f"z{i + 1}", kind, tdtype), rtol=rtol, atol=atol)

    if not backward:
        return
    loss = sum((z * array(m, f"dldz{i + 1}", k, tdtype)).sum() for i, (z, k) in enumerate(zip(Z, out_kinds)))
    loss.backward()
    rtol, atol = tolerance(dtype, gradient=True)
    for i, (x, kind) in enumerate(zip(X, in_kinds)):
        torch.testing.assert_close(x.grad, array(m, f"dldx{i + 1}", kind, tdtype), rtol=rtol, atol=atol)
    if "dldw" in m:
        expected = torch.as_tensor(m["dldw"].ravel(), dtype=torch.float64)
        grad = layer.angles.grad if layer.angles.grad is not None else torch.zeros_like(layer.angles)
        torch.testing.assert_close(grad, expected, rtol=rtol, atol=atol)
