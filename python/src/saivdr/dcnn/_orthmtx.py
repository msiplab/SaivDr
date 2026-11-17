"""Orthonormal matrix generation by Givens rotations.

Equivalent to ``saivdr.dcnn.fcn_orthmtxgen`` in MATLAB. The generated
matrix is ``diag(mus) @ G_K @ ... @ G_2 @ G_1``, where ``G_k`` rotates
the pair of rows ``(iTop, iBtm)`` in the order (0,1), (0,2), ...,
(0,n-1), (1,2), ... with ``angles[k]``.

Gradients are obtained by autograd, which coincides with the custom
backward functions of the MATLAB layers.
"""

import math

import torch

from .exceptions import InvalidMus

__all__ = ["number_of_dimensions", "number_of_angles", "orthmtxgen", "check_mus"]


def number_of_angles(n):
    """Number of Givens rotation angles for an ``n x n`` matrix."""
    return n * (n - 1) // 2


def number_of_dimensions(n_angles):
    """Matrix size ``n`` for ``n_angles`` angles, as in MATLAB."""
    n = (1 + math.isqrt(1 + 8 * n_angles)) // 2
    if number_of_angles(n) != n_angles:
        raise ValueError(f"{n_angles} is not a valid number of angles")
    return n


def check_mus(mus):
    """Raise :class:`InvalidMus` unless every element of ``mus`` is +1 or -1."""
    if not bool(torch.all(torch.abs(mus) == 1)):
        raise InvalidMus(f"{mus.tolist()} : Elements in mus should be either of 1 or -1")


def orthmtxgen(angles, mus):
    """Generate an orthonormal matrix from ``angles`` and ``mus``.

    Parameters
    ----------
    angles : torch.Tensor
        1-D tensor of ``n(n-1)/2`` rotation angles.
    mus : torch.Tensor or float
        Signs (+1/-1) applied to the rows, of length ``n`` or a scalar.

    Returns
    -------
    torch.Tensor
        ``n x n`` matrix of the same dtype and device as ``angles``.
    """
    n = number_of_dimensions(angles.numel())
    rows = list(torch.eye(n, dtype=angles.dtype, device=angles.device).unbind(0))
    iAng = 0
    for iTop in range(n - 1):
        vt = rows[iTop]
        for iBtm in range(iTop + 1, n):
            angle = angles[iAng]
            c = torch.cos(angle)
            s = torch.sin(angle)
            vb = rows[iBtm]
            # The same arithmetic as fcn_orthmtxgen.m
            u = s * (vt + vb)
            vt = (c + s) * vt - u
            rows[iBtm] = (c - s) * vb + u
            iAng += 1
        rows[iTop] = vt
    matrix = torch.stack(rows, dim=0)
    if not torch.is_tensor(mus):
        mus = torch.full((n,), float(mus), dtype=angles.dtype, device=angles.device)
    return mus.to(dtype=angles.dtype, device=angles.device).reshape(-1, 1) * matrix
