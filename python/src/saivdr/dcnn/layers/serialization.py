"""Subband (de)serialization layers.

MATLAB nsoltSubbandSerialization2d/3dLayer and
nsoltSubbandDeserialization2d/3dLayer.

The serialized vector of each sample consists of the DC subband of the
last level followed by the AC subbands of levels ``nLevels, ..., 1``. Each
subband is vectorized in the column-major order of MATLAB, i.e. over
``(nRows, nCols, nChs)`` in 2-D and ``(nRows, nCols, nLays, nChs)`` in 3-D,
so that the serialized vectors are identical to the MATLAB ones.
"""

import math

import torch
import torch.nn as nn

from ..utility import ChannelGroup
from ._common import as_pair

__all__ = [
    "NsoltSubbandSerialization2dLayer",
    "NsoltSubbandDeserialization2dLayer",
    "NsoltSubbandSerialization3dLayer",
    "NsoltSubbandDeserialization3dLayer",
]


class _SubbandBase(nn.Module):
    _ndims = 2

    def __init__(self, original_dimension, number_of_channels, decimation_factor, number_of_levels, name):
        super().__init__()
        self.name = name
        self.number_of_channels = as_pair(number_of_channels)
        self.decimation_factor = tuple(int(v) for v in decimation_factor)
        self.number_of_levels = int(number_of_levels)
        self.set_original_dimension(original_dimension)

    def set_original_dimension(self, original_dimension):
        """Set ``original_dimension`` (MATLAB order) and update ``scales``.

        ``scales[0]`` is the size of the DC subband of the last level and
        ``scales[i]`` that of the AC subbands of level ``nLevels-i+1``, both in
        the MATLAB order ``(nRows, nCols, [nLays,] nChs)``.
        """
        self.original_dimension = tuple(int(v) for v in original_dimension)
        nLevels = self.number_of_levels
        nChsTotal = sum(self.number_of_channels)
        stride = self.decimation_factor
        base = []
        for size, factor in zip(self.original_dimension, stride):
            if size % factor**nLevels != 0:
                raise ValueError(f"{self.original_dimension} is not divisible by {stride}^{nLevels}")
            base.append(size // factor**nLevels)
        scales = [tuple(base) + (1,)]
        for iRevLv in range(1, nLevels + 1):
            scales.append(tuple(b * f ** (iRevLv - 1) for b, f in zip(base, stride)) + (nChsTotal - 1,))
        self.scales = tuple(scales)
        ps, pa = self.number_of_channels[ChannelGroup.UPPER], self.number_of_channels[ChannelGroup.LOWER]
        dims = ",".join(str(v) for v in self.original_dimension)
        decs = ",".join(str(v) for v in stride)
        self.description = (
            f"Subband {self._kind} ({dims}), lv = {nLevels}, (ps,pa) = ({ps},{pa}), dec = ({decs})"
        )

    @property
    def number_of_elements(self):
        return sum(math.prod(s) for s in self.scales)

    # Column-major vectorization of nSamples x nChs x <spatial>
    def _to_vector(self, X):
        if self._ndims == 2:  # (N, C, H, W) -> order (C, W, H)
            Y = X.permute(0, 1, 3, 2)
        else:  # (N, C, D, H, W) -> order (C, D, W, H)
            Y = X.permute(0, 1, 2, 4, 3)
        return Y.reshape(X.size(0), -1)

    def _from_vector(self, x, scale):
        nSamples = x.size(0)
        if self._ndims == 2:
            nrows, ncols, nchs = scale
            return x.reshape(nSamples, nchs, ncols, nrows).permute(0, 1, 3, 2)
        nrows, ncols, nlays, nchs = scale
        return x.reshape(nSamples, nchs, nlays, ncols, nrows).permute(0, 1, 2, 4, 3)


class _Serialization(_SubbandBase):
    _kind = "serialization"

    def forward(self, *inputs):
        """Serialize ``(Lv1_Ac, ..., LvN_Ac, LvN_Dc)`` into ``nSamples x nElements``."""
        nLevels = self.number_of_levels
        if len(inputs) != nLevels + 1:
            raise ValueError(f"{nLevels + 1} inputs are expected, but {len(inputs)} were given")
        parts = [self._to_vector(inputs[nLevels])]
        for iRevLv in range(1, nLevels + 1):
            parts.append(self._to_vector(inputs[nLevels - iRevLv]))
        return torch.cat(parts, dim=1)


class _Deserialization(_SubbandBase):
    _kind = "deserialization"

    @property
    def input_size(self):
        return (self.number_of_elements,)

    def forward(self, X):
        """Deserialize ``nSamples x nElements`` into ``(Lv1_Ac, ..., LvN_Ac, LvN_Dc)``."""
        nLevels = self.number_of_levels
        X = X.reshape(X.size(0), -1)
        outputs = [None] * (nLevels + 1)
        sidx = 0
        for iRevLv, scale in enumerate(self.scales):
            n = math.prod(scale)
            sub = self._from_vector(X[:, sidx : sidx + n], scale)
            outputs[nLevels if iRevLv == 0 else nLevels - iRevLv] = sub
            sidx += n
        return tuple(outputs)


class NsoltSubbandSerialization2dLayer(_Serialization):
    """Equivalent to MATLAB ``nsoltSubbandSerialization2dLayer``.

    Inputs: ``Lv1_Ac, ..., LvN_Ac`` (``nSamples x (P-1) x nRows x nCols``) and
    ``LvN_Dc`` (``nSamples x 1 x nRows x nCols``). Output: ``nSamples x nElements``.
    ``original_dimension`` is ``(height, width)``.
    """

    _ndims = 2

    def __init__(
        self,
        original_dimension=(8, 8),
        number_of_channels=(2, 2),
        decimation_factor=(2, 2),
        number_of_levels=1,
        name="",
    ):
        super().__init__(original_dimension, number_of_channels, decimation_factor, number_of_levels, name)


class NsoltSubbandDeserialization2dLayer(_Deserialization):
    """Equivalent to MATLAB ``nsoltSubbandDeserialization2dLayer`` (inverse of serialization)."""

    _ndims = 2

    def __init__(
        self,
        original_dimension=(8, 8),
        number_of_channels=(2, 2),
        decimation_factor=(2, 2),
        number_of_levels=1,
        name="",
    ):
        super().__init__(original_dimension, number_of_channels, decimation_factor, number_of_levels, name)


class NsoltSubbandSerialization3dLayer(_Serialization):
    """Equivalent to MATLAB ``nsoltSubbandSerialization3dLayer``.

    Inputs are ``nSamples x nChs x nLays x nRows x nCols`` arrays.
    ``original_dimension`` is ``(height, width, depth)`` as in MATLAB.
    """

    _ndims = 3

    def __init__(
        self,
        original_dimension=(8, 8, 8),
        number_of_channels=(4, 4),
        decimation_factor=(2, 2, 2),
        number_of_levels=1,
        name="",
    ):
        super().__init__(original_dimension, number_of_channels, decimation_factor, number_of_levels, name)


class NsoltSubbandDeserialization3dLayer(_Deserialization):
    """Equivalent to MATLAB ``nsoltSubbandDeserialization3dLayer``."""

    _ndims = 3

    def __init__(
        self,
        original_dimension=(8, 8, 8),
        number_of_channels=(4, 4),
        decimation_factor=(2, 2, 2),
        number_of_levels=1,
        name="",
    ):
        super().__init__(original_dimension, number_of_channels, decimation_factor, number_of_levels, name)
