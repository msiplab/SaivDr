"""Block DCT and IDCT layers (MATLAB nsoltBlockDct2d/3dLayer, nsoltBlockIdct2d/3dLayer)."""

import torch
import torch.nn as nn

from .._dct import block_dct_matrix_2d, block_dct_matrix_3d
from ..utility import Direction

__all__ = [
    "NsoltBlockDct2dLayer",
    "NsoltBlockIdct2dLayer",
    "NsoltBlockDct3dLayer",
    "NsoltBlockIdct3dLayer",
]


def _split(size, factor, what):
    if size % factor != 0:
        raise ValueError(f"The {what} size {size} is not divisible by the decimation factor {factor}")
    return size // factor


class NsoltBlockDct2dLayer(nn.Module):
    """Block DCT layer, equivalent to MATLAB ``nsoltBlockDct2dLayer``.

    Input: ``nSamples x nComponents x (decV*nRows) x (decH*nCols)``.

    Output: for each component, ``nSamples x nDecs x nRows x nCols``. A tensor
    is returned for one component and a tuple of tensors otherwise.
    """

    def __init__(self, decimation_factor=(2, 2), number_of_components=1, name=""):
        super().__init__()
        self.decimation_factor = tuple(int(v) for v in decimation_factor)
        self.number_of_components = int(number_of_components)
        self.name = name
        decV, decH = self.decimation_factor
        self.description = f"Block DCT of size {decV}x{decH}"

    def forward(self, X):
        decV = self.decimation_factor[Direction.VERTICAL]
        decH = self.decimation_factor[Direction.HORIZONTAL]
        nSamples, nComponents, height, width = X.shape
        nrows = _split(height, decV, "vertical")
        ncols = _split(width, decH, "horizontal")
        Cvh = block_dct_matrix_2d(self.decimation_factor, dtype=X.dtype, device=X.device)
        # Vectorize each block in column-major order (v + decV*h)
        Y = (
            X.reshape(nSamples, nComponents, nrows, decV, ncols, decH)
            .permute(0, 1, 2, 4, 5, 3)
            .reshape(nSamples, nComponents, nrows, ncols, decV * decH)
        )
        Z = torch.movedim(Y @ Cvh.T, -1, 2)  # nSamples x nComponents x nDecs x nRows x nCols
        outputs = tuple(Z[:, iCmp] for iCmp in range(nComponents))
        return outputs[0] if self.number_of_components == 1 else outputs


class NsoltBlockIdct2dLayer(nn.Module):
    """Block IDCT layer, equivalent to MATLAB ``nsoltBlockIdct2dLayer``.

    Input: one ``nSamples x nDecs x nRows x nCols`` tensor per component.

    Output: ``nSamples x nComponents x (decV*nRows) x (decH*nCols)``.
    """

    def __init__(self, decimation_factor=(2, 2), number_of_components=1, name=""):
        super().__init__()
        self.decimation_factor = tuple(int(v) for v in decimation_factor)
        self.number_of_components = int(number_of_components)
        self.name = name
        decV, decH = self.decimation_factor
        self.description = f"Block IDCT of size {decV}x{decH}"

    def forward(self, *inputs):
        if len(inputs) != self.number_of_components:
            raise ValueError(f"{self.number_of_components} inputs are expected, but {len(inputs)} were given")
        decV = self.decimation_factor[Direction.VERTICAL]
        decH = self.decimation_factor[Direction.HORIZONTAL]
        outputs = []
        for X in inputs:
            nSamples, _, nrows, ncols = X.shape
            Cvh = block_dct_matrix_2d(self.decimation_factor, dtype=X.dtype, device=X.device)
            Y = torch.movedim(X, 1, -1) @ Cvh  # nSamples x nRows x nCols x (decH*decV)
            Y = (
                Y.reshape(nSamples, nrows, ncols, decH, decV)
                .permute(0, 1, 4, 2, 3)
                .reshape(nSamples, nrows * decV, ncols * decH)
            )
            outputs.append(Y)
        return torch.stack(outputs, dim=1)


class NsoltBlockDct3dLayer(nn.Module):
    """Block DCT layer, equivalent to MATLAB ``nsoltBlockDct3dLayer``.

    Input: ``nSamples x nComponents x (decD*nLays) x (decV*nRows) x (decH*nCols)``.

    Output: for each component, ``nSamples x nDecs x nLays x nRows x nCols``.
    ``decimation_factor`` is given in the MATLAB order ``(decV, decH, decD)``.
    """

    def __init__(self, decimation_factor=(2, 2, 2), number_of_components=1, name=""):
        super().__init__()
        self.decimation_factor = tuple(int(v) for v in decimation_factor)
        self.number_of_components = int(number_of_components)
        self.name = name
        decV, decH, decD = self.decimation_factor
        self.description = f"Block DCT of size {decV}x{decH}x{decD}"

    def forward(self, X):
        decV, decH, decD = self.decimation_factor
        nSamples, nComponents, depth, height, width = X.shape
        nlays = _split(depth, decD, "depth")
        nrows = _split(height, decV, "vertical")
        ncols = _split(width, decH, "horizontal")
        Cvhd = block_dct_matrix_3d(self.decimation_factor, dtype=X.dtype, device=X.device)
        # Vectorize each block in column-major order (v + decV*h + decV*decH*d)
        Y = (
            X.reshape(nSamples, nComponents, nlays, decD, nrows, decV, ncols, decH)
            .permute(0, 1, 2, 4, 6, 3, 7, 5)
            .reshape(nSamples, nComponents, nlays, nrows, ncols, decD * decH * decV)
        )
        Z = torch.movedim(Y @ Cvhd.T, -1, 2)
        outputs = tuple(Z[:, iCmp] for iCmp in range(nComponents))
        return outputs[0] if self.number_of_components == 1 else outputs


class NsoltBlockIdct3dLayer(nn.Module):
    """Block IDCT layer, equivalent to MATLAB ``nsoltBlockIdct3dLayer``.

    Input: one ``nSamples x nDecs x nLays x nRows x nCols`` tensor per component.

    Output: ``nSamples x nComponents x (decD*nLays) x (decV*nRows) x (decH*nCols)``.
    """

    def __init__(self, decimation_factor=(2, 2, 2), number_of_components=1, name=""):
        super().__init__()
        self.decimation_factor = tuple(int(v) for v in decimation_factor)
        self.number_of_components = int(number_of_components)
        self.name = name
        decV, decH, decD = self.decimation_factor
        self.description = f"Block IDCT of size {decV}x{decH}x{decD}"

    def forward(self, *inputs):
        if len(inputs) != self.number_of_components:
            raise ValueError(f"{self.number_of_components} inputs are expected, but {len(inputs)} were given")
        decV, decH, decD = self.decimation_factor
        outputs = []
        for X in inputs:
            nSamples, _, nlays, nrows, ncols = X.shape
            Cvhd = block_dct_matrix_3d(self.decimation_factor, dtype=X.dtype, device=X.device)
            Y = torch.movedim(X, 1, -1) @ Cvhd  # ... x (decD*decH*decV)
            Y = (
                Y.reshape(nSamples, nlays, nrows, ncols, decD, decH, decV)
                .permute(0, 1, 4, 2, 6, 3, 5)
                .reshape(nSamples, nlays * decD, nrows * decV, ncols * decH)
            )
            outputs.append(Y)
        return torch.stack(outputs, dim=1)
