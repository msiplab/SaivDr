"""Conversion between the MATLAB and PyTorch implementations.

Array layouts
-------------
=========  =============================  ==================================
kind       MATLAB                         PyTorch
=========  =============================  ==================================
image2d    height x width x nCmps x N     N x nCmps x height x width
coef2d     nChs x nRows x nCols x N       N x nChs x nRows x nCols
image3d    H x W x D x nCmps x N          N x nCmps x D x H x W
coef3d     nChs x nRows x nCols x nLays   N x nChs x nLays x nRows x nCols
           x N
serial     nElements x 1 x 1 [x 1] x N    N x nElements
=========  =============================  ==================================

``image2d``/``image3d`` are also used for the subbands passed between
``nsoltChannelSeparation*`` and ``nsoltChannelConcatenation*`` layers and
for the network inputs and outputs.

Parameters
----------
The ``angles`` and ``mus`` of the PyTorch rotation layers are identical to
the ``Angles`` and ``Mus`` properties of the MATLAB layers, and the layers
of the networks are keyed by the MATLAB layer names. :func:`save_params`
and :func:`load_params` exchange them through a MAT-file that the MATLAB
functions ``saivdr_export_params`` and ``saivdr_import_params`` (in
``python/tools/matlab``) read and write.
"""

import numpy as np
import torch

__all__ = [
    "from_matlab_array",
    "to_matlab_array",
    "rotation_layers",
    "get_params",
    "set_params",
    "save_params",
    "load_params",
]

# MATLAB axis -> PyTorch axis, for each kind of array
_PERMUTATIONS = {
    "image2d": (3, 2, 0, 1),
    "coef2d": (3, 0, 1, 2),
    "image3d": (4, 3, 2, 0, 1),
    "coef3d": (4, 0, 3, 1, 2),
}
_NDIMS = {"image2d": 4, "coef2d": 4, "image3d": 5, "coef3d": 5}


def _pad_dims(a, ndims):
    a = np.asarray(a)
    if a.ndim > ndims:
        raise ValueError(f"An array of at most {ndims} dimensions is expected, but got {a.shape}")
    return a.reshape(a.shape + (1,) * (ndims - a.ndim))


def from_matlab_array(a, kind, size=None, dtype=None, device=None):
    """Convert a MATLAB array (e.g. read by ``scipy.io.loadmat``) to a tensor.

    ``size`` restores the trailing singleton dimensions that MATLAB drops;
    without it, they are inferred from ``kind``.
    """
    a = np.asarray(a)
    if size is not None:
        a = a.reshape(tuple(int(s) for s in np.ravel(size)), order="F")
    if kind == "serial":
        a = a.reshape(a.shape[0], -1, order="F")  # nElements x N
        a = a.T
    else:
        a = _pad_dims(a, _NDIMS[kind]).transpose(_PERMUTATIONS[kind])
    t = torch.from_numpy(np.ascontiguousarray(a))
    return t.to(dtype=dtype, device=device) if (dtype or device) else t


def to_matlab_array(t, kind):
    """Convert a tensor to a NumPy array in the MATLAB layout."""
    a = t.detach().cpu().numpy()
    if kind == "serial":
        return np.ascontiguousarray(a.T.reshape(a.shape[1], 1, 1, a.shape[0]))
    inverse = np.argsort(_PERMUTATIONS[kind])
    return np.ascontiguousarray(a.transpose(inverse))


def rotation_layers(net):
    """Return ``{MATLAB layer name: layer}`` of the rotation layers of ``net``."""
    from .layers.rotation import _RotationBase

    layers = net.layers if hasattr(net, "layers") else {}
    return {name: layer for name, layer in layers.items() if isinstance(layer, _RotationBase)}


def get_params(net):
    """Return ``{name: {'Angles', 'Mus'[, 'NoDcLeakage']}}`` as NumPy arrays."""
    params = {}
    for name, layer in rotation_layers(net).items():
        p = {
            "Angles": layer.angles.detach().cpu().double().numpy().copy(),
            "Mus": layer.mus.detach().cpu().double().numpy().copy(),
        }
        if hasattr(layer, "no_dc_leakage"):
            p["NoDcLeakage"] = bool(layer.no_dc_leakage)
        params[name] = p
    return params


def set_params(net, params, strict=True):
    """Set the parameters returned by :func:`get_params` or read from MATLAB."""
    layers = rotation_layers(net)
    missing = set(layers) - set(params)
    unexpected = set(params) - set(layers)
    if strict and (missing or unexpected):
        raise KeyError(f"missing: {sorted(missing)}, unexpected: {sorted(unexpected)}")
    for name, p in params.items():
        if name not in layers:
            continue
        layer = layers[name]
        if "NoDcLeakage" in p and hasattr(layer, "no_dc_leakage"):
            layer.no_dc_leakage = bool(np.asarray(p["NoDcLeakage"]).item())
        layer.set_angles(np.asarray(p["Angles"], dtype=np.float64).reshape(-1))
        layer.set_mus(np.asarray(p["Mus"], dtype=np.float64).reshape(-1))
    return net


def save_params(net, path):
    """Save the parameters of ``net`` to a MAT-file readable by ``saivdr_import_params``."""
    from scipy.io import savemat

    params = get_params(net)
    names = list(params)
    cell = lambda values: np.array(values + [None], dtype=object)[:-1].reshape(-1, 1)  # noqa: E731
    savemat(
        path,
        {
            "names": cell([np.array(n) for n in names]),
            "angles": cell([params[n]["Angles"].reshape(-1, 1) for n in names]),
            "mus": cell([params[n]["Mus"].reshape(-1, 1) for n in names]),
            "nodcleakage": np.array([float(params[n].get("NoDcLeakage", np.nan)) for n in names]).reshape(
                -1, 1
            ),
        },
    )


def load_params(net, path, strict=True):
    """Load parameters saved by ``saivdr_export_params`` in MATLAB (or :func:`save_params`)."""
    from scipy.io import loadmat

    m = loadmat(path)
    names = [
        str(np.asarray(n).item() if np.asarray(n).size == 1 else "".join(np.asarray(n).ravel()))
        for n in m["names"].ravel()
    ]
    params = {}
    for i, name in enumerate(names):
        p = {
            "Angles": np.asarray(m["angles"].ravel()[i], dtype=np.float64).reshape(-1),
            "Mus": np.asarray(m["mus"].ravel()[i], dtype=np.float64).reshape(-1),
        }
        ndl = float(np.asarray(m["nodcleakage"]).ravel()[i])
        if not np.isnan(ndl):
            p["NoDcLeakage"] = bool(ndl)
        params[name] = p
    return set_params(net, params, strict=strict)
