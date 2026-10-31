"""Rotation layers (MATLAB nsoltInitial/Final/IntermediateRotation2d/3dLayer).

The learnable ``angles`` and the buffer ``mus`` are laid out exactly as the
``Angles`` and ``Mus`` properties of the MATLAB layers, so they can be
exchanged with MATLAB without any conversion.
"""

import math

import torch
import torch.nn as nn

from .._orthmtx import check_mus, orthmtxgen
from ..exceptions import InvalidMode, InvalidNumberOfAngles
from ._common import apply_channel_matrix, as_pair

__all__ = [
    "NsoltInitialRotation2dLayer",
    "NsoltInitialRotation3dLayer",
    "NsoltFinalRotation2dLayer",
    "NsoltFinalRotation3dLayer",
    "NsoltIntermediateRotation2dLayer",
    "NsoltIntermediateRotation3dLayer",
]


def _vector(value, length, default, dtype, device, what):
    """Expand ``value`` to a 1-D tensor as the MATLAB property setters do."""
    if value is None:
        return torch.full((length,), float(default), dtype=dtype, device=device)
    t = torch.as_tensor(value, dtype=dtype, device=device).reshape(-1).clone()
    if t.numel() == 1 and length != 1:
        return t.expand(length).clone()
    if t.numel() != length:
        raise InvalidNumberOfAngles(f"{length} {what} are expected, but {t.numel()} were given")
    return t


class _RotationBase(nn.Module):
    """Common part of the rotation layers.

    ``angles`` is created in ``dtype`` (``torch.get_default_dtype()`` when
    ``None``). Matrices are always generated in float64 and then cast to the
    dtype of the input, as MATLAB does with double-precision angles.
    """

    def __init__(self, number_of_channels, n_angles, n_mus, angles, mus, name, dtype, device):
        super().__init__()
        self.number_of_channels = as_pair(number_of_channels)
        self.name = name
        dtype = torch.get_default_dtype() if dtype is None else dtype
        self.angles = nn.Parameter(_vector(angles, n_angles, 0.0, dtype, device, "angles"))
        mus_ = _vector(mus, n_mus, 1.0, dtype, device, "mus")
        check_mus(mus_)
        self.register_buffer("mus", mus_)

    def set_mus(self, mus):
        """Replace ``mus`` (a scalar is expanded as in MATLAB)."""
        mus_ = _vector(mus, self.mus.numel(), 1.0, self.mus.dtype, self.mus.device, "mus")
        check_mus(mus_)
        self.mus.copy_(mus_)

    def set_angles(self, angles):
        """Replace ``angles`` (a scalar is expanded as in MATLAB)."""
        a = self.angles
        with torch.no_grad():
            a.copy_(_vector(angles, a.numel(), 0.0, a.dtype, a.device, "angles"))


class _InitialFinalBase(_RotationBase):
    def __init__(
        self, number_of_channels, decimation_factor, no_dc_leakage, angles, mus, name, dtype, device
    ):
        ps, pa = as_pair(number_of_channels)
        nChsTotal = ps + pa
        nAngles = (nChsTotal - 2) * nChsTotal // 4
        super().__init__(number_of_channels, nAngles, nChsTotal, angles, mus, name, dtype, device)
        self.decimation_factor = tuple(int(v) for v in decimation_factor)
        self.no_dc_leakage = bool(no_dc_leakage)
        if self.no_dc_leakage:
            # As updateParameters in MATLAB, which also overwrites the
            # stored properties
            with torch.no_grad():
                self.mus[0] = 1
                self.angles[: ps - 1] = 0

    def matrices(self):
        """Return ``(W0, U0)`` in float64, as ``updateParameters`` in MATLAB.

        When ``no_dc_leakage`` is set, ``mus[0]`` is regarded as 1 and the
        first ``ps-1`` angles as 0. As in the MATLAB backward functions, the
        gradients with respect to those angles are not zeroed.
        """
        ps, _ = self.number_of_channels
        angles = self.angles.to(torch.float64)
        mus = self.mus.to(torch.float64)
        if self.no_dc_leakage:
            mask = torch.zeros_like(angles, dtype=torch.bool)
            mask[: ps - 1] = True
            # Zero in value, but the gradient passes through
            angles = torch.where(mask, angles - angles.detach(), angles)
            mus = torch.cat((torch.ones_like(mus[:1]), mus[1:]))
        nAngles = angles.numel()
        W0 = orthmtxgen(angles[: nAngles // 2], mus[:ps])
        U0 = orthmtxgen(angles[nAngles // 2 :], mus[ps:])
        return W0, U0

    def _description(self, title):
        ps, pa = self.number_of_channels
        names = ("mv", "mh", "md")[: len(self.decimation_factor)]
        return (
            f"NSOLT {title} rotation (ps,pa) = ({ps},{pa}), "
            f"({','.join(names)}) = ({','.join(str(v) for v in self.decimation_factor)})"
        )


class _InitialRotation(_InitialFinalBase):
    def forward(self, X):
        nDecs = math.prod(self.decimation_factor)
        ms, ma = (nDecs + 1) // 2, nDecs // 2
        W0, U0 = self.matrices()
        Zs = apply_channel_matrix(W0[:, :ms], X[:, :ms])
        Za = apply_channel_matrix(U0[:, :ma], X[:, ms:nDecs])
        return torch.cat((Zs, Za), dim=1)


class _FinalRotation(_InitialFinalBase):
    def forward(self, X):
        ps, pa = self.number_of_channels
        nDecs = math.prod(self.decimation_factor)
        ms, ma = (nDecs + 1) // 2, nDecs // 2
        W0, U0 = self.matrices()
        Zs = apply_channel_matrix(W0.T[:ms, :], X[:, :ps])
        Za = apply_channel_matrix(U0.T[:ma, :], X[:, ps : ps + pa])
        return torch.cat((Zs, Za), dim=1)


class NsoltInitialRotation2dLayer(_InitialRotation):
    """Initial rotation layer, equivalent to MATLAB ``nsoltInitialRotation2dLayer``.

    Input: ``nSamples x nDecs x nRows x nCols``.
    Output: ``nSamples x (ps+pa) x nRows x nCols``.

    ``angles`` has ``(P-2)P/4`` elements (``P = ps+pa``) ordered as
    ``[angles of W0; angles of U0]`` and ``mus`` is ``[mus of W0; mus of U0]``.
    """

    def __init__(
        self,
        number_of_channels,
        decimation_factor,
        no_dc_leakage=False,
        name="",
        angles=None,
        mus=None,
        dtype=None,
        device=None,
    ):
        super().__init__(
            number_of_channels, decimation_factor, no_dc_leakage, angles, mus, name, dtype, device
        )
        self.description = self._description("initial")


class NsoltInitialRotation3dLayer(_InitialRotation):
    """Initial rotation layer, equivalent to MATLAB ``nsoltInitialRotation3dLayer``.

    Input: ``nSamples x nDecs x nLays x nRows x nCols``.
    Output: ``nSamples x (ps+pa) x nLays x nRows x nCols``.
    """

    def __init__(
        self,
        number_of_channels,
        decimation_factor,
        no_dc_leakage=False,
        name="",
        angles=None,
        mus=None,
        dtype=None,
        device=None,
    ):
        super().__init__(
            number_of_channels, decimation_factor, no_dc_leakage, angles, mus, name, dtype, device
        )
        self.description = self._description("initial")


class NsoltFinalRotation2dLayer(_FinalRotation):
    """Final rotation layer, equivalent to MATLAB ``nsoltFinalRotation2dLayer``.

    Input: ``nSamples x (ps+pa) x nRows x nCols``.
    Output: ``nSamples x nDecs x nRows x nCols``.
    """

    def __init__(
        self,
        number_of_channels,
        decimation_factor,
        no_dc_leakage=False,
        name="",
        angles=None,
        mus=None,
        dtype=None,
        device=None,
    ):
        super().__init__(
            number_of_channels, decimation_factor, no_dc_leakage, angles, mus, name, dtype, device
        )
        self.description = self._description("final")


class NsoltFinalRotation3dLayer(_FinalRotation):
    """Final rotation layer, equivalent to MATLAB ``nsoltFinalRotation3dLayer``.

    Input: ``nSamples x (ps+pa) x nLays x nRows x nCols``.
    Output: ``nSamples x nDecs x nLays x nRows x nCols``.
    """

    def __init__(
        self,
        number_of_channels,
        decimation_factor,
        no_dc_leakage=False,
        name="",
        angles=None,
        mus=None,
        dtype=None,
        device=None,
    ):
        super().__init__(
            number_of_channels, decimation_factor, no_dc_leakage, angles, mus, name, dtype, device
        )
        self.description = self._description("final")


class _IntermediateRotation(_RotationBase):
    def __init__(self, number_of_channels, mode, name, angles, mus, dtype, device):
        ps, pa = as_pair(number_of_channels)
        nChsTotal = ps + pa
        nAngles = (nChsTotal - 2) * nChsTotal // 8
        super().__init__(number_of_channels, nAngles, pa, angles, mus, name, dtype, device)
        self.mode = mode
        self.description = f"{mode} NSOLT intermediate rotation (ps,pa) = ({ps},{pa})"

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode not in ("Analysis", "Synthesis"):
            raise InvalidMode(f"{mode} : Mode should be either of Synthesis or Analysis")
        self._mode = mode

    def matrix(self):
        """Return ``Un`` in float64."""
        return orthmtxgen(self.angles.to(torch.float64), self.mus.to(torch.float64))

    def forward(self, X):
        ps, pa = self.number_of_channels
        Un = self.matrix()
        if self.mode == "Synthesis":
            Un = Un.T
        Za = apply_channel_matrix(Un, X[:, ps : ps + pa])
        return torch.cat((X[:, :ps], Za), dim=1)


class NsoltIntermediateRotation2dLayer(_IntermediateRotation):
    """Intermediate rotation layer, equivalent to MATLAB ``nsoltIntermediateRotation2dLayer``.

    Input and output: ``nSamples x (ps+pa) x nRows x nCols``. Only the lower
    ``pa`` channels are rotated, by ``Un`` (Analysis) or ``Un.T`` (Synthesis).
    """

    def __init__(
        self, number_of_channels, mode="Synthesis", name="", mus=None, angles=None, dtype=None, device=None
    ):
        super().__init__(number_of_channels, mode, name, angles, mus, dtype, device)


class NsoltIntermediateRotation3dLayer(_IntermediateRotation):
    """Intermediate rotation layer, equivalent to MATLAB ``nsoltIntermediateRotation3dLayer``.

    Input and output: ``nSamples x (ps+pa) x nLays x nRows x nCols``.
    """

    def __init__(
        self, number_of_channels, mode="Synthesis", name="", mus=None, angles=None, dtype=None, device=None
    ):
        super().__init__(number_of_channels, mode, name, angles, mus, dtype, device)
