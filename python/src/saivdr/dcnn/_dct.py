"""Block DCT matrices equivalent to those in MATLAB nsoltBlockDct2d/3dLayer."""

import functools
import math

import torch

from .utility import Direction

__all__ = ["dctmtx", "block_dct_matrix_2d", "block_dct_matrix_3d"]


def dctmtx(n, dtype=torch.float64, device=None):
    """Orthonormal DCT-II matrix, equivalent to MATLAB ``dctmtx(n)``."""
    k = torch.arange(n, dtype=torch.float64).reshape(-1, 1)
    i = torch.arange(n, dtype=torch.float64).reshape(1, -1)
    C = math.sqrt(2.0 / n) * torch.cos(math.pi * (2 * i + 1) * k / (2 * n))
    C[0, :] = math.sqrt(1.0 / n)
    return C.to(dtype=dtype, device=device)


def _dctmtx_even_odd(n):
    C = dctmtx(n)
    return C[0::2, :], C[1::2, :]


@functools.cache
def _block_dct_matrix_2d(decV, decH):
    Cve, Cvo = _dctmtx_even_odd(decV)
    Che, Cho = _dctmtx_even_odd(decH)
    Cee = torch.kron(Che, Cve)
    Coo = torch.kron(Cho, Cvo)
    Coe = torch.kron(Che, Cvo)
    Ceo = torch.kron(Cho, Cve)
    return torch.cat((Cee, Coo, Coe, Ceo), dim=0)


@functools.cache
def _block_dct_matrix_3d(decV, decH, decD):
    Cve, Cvo = _dctmtx_even_odd(decV)
    Che, Cho = _dctmtx_even_odd(decH)
    Cde, Cdo = _dctmtx_even_odd(decD)
    groups = (
        (Cve, Che, Cde),  # Ceee
        (Cve, Cho, Cdo),  # Ceoo
        (Cvo, Cho, Cde),  # Cooe
        (Cvo, Che, Cdo),  # Coeo
        (Cve, Che, Cdo),  # Ceeo
        (Cve, Cho, Cde),  # Ceoe
        (Cvo, Cho, Cdo),  # Cooo
        (Cvo, Che, Cde),  # Coee
    )
    ncols = decV * decH * decD
    rows = []
    for Cy, Cx, Cz in groups:
        # kron rows are in the order (z, x, y) with y fastest
        G = torch.kron(Cz, torch.kron(Cx, Cy))
        ny, nx, nz = Cy.size(0), Cx.size(0), Cz.size(0)
        # Reorder to y slowest and z fastest as in permuteRowsYXZ_
        G = G.reshape(nz, nx, ny, ncols).permute(2, 1, 0, 3).reshape(ny * nx * nz, ncols)
        rows.append(G)
    return torch.cat(rows, dim=0)


def block_dct_matrix_2d(decimation_factor, dtype=torch.float64, device=None):
    """2-D block DCT matrix ``Cvh`` of MATLAB nsoltBlockDct2dLayer.

    Rows are ordered as ``[Cee; Coo; Coe; Ceo]`` and columns correspond to
    the pixels of a ``decV x decH`` block in column-major order
    (``v + decV*h``).
    """
    decV = int(decimation_factor[Direction.VERTICAL])
    decH = int(decimation_factor[Direction.HORIZONTAL])
    return _block_dct_matrix_2d(decV, decH).to(dtype=dtype, device=device)


def block_dct_matrix_3d(decimation_factor, dtype=torch.float64, device=None):
    """3-D block DCT matrix ``Cvhd`` of MATLAB nsoltBlockDct3dLayer.

    Rows are ordered as ``[Ceee; Ceoo; Cooe; Coeo; Ceeo; Ceoe; Cooo; Coee]``,
    where the coefficients in each group are ordered as in
    ``saivdr.dictionary.nsoltx`` (depth fastest, vertical slowest). Columns
    correspond to the voxels of a ``decV x decH x decD`` block in
    column-major order (``v + decV*h + decV*decH*d``).
    """
    decV = int(decimation_factor[Direction.VERTICAL])
    decH = int(decimation_factor[Direction.HORIZONTAL])
    decD = int(decimation_factor[Direction.DEPTH])
    return _block_dct_matrix_3d(decV, decH, decD).to(dtype=dtype, device=device)
