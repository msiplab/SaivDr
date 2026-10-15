"""Atom extension layers (MATLAB nsoltAtomExtension2d/3dLayer)."""

import torch
import torch.nn as nn

from ..exceptions import InvalidDirection, InvalidTargetChannels
from ._common import as_pair

__all__ = ["NsoltAtomExtension2dLayer", "NsoltAtomExtension3dLayer"]

# (dimension counted from the last one, shift) of the coefficient arrays
# nSamples x nChs x [nLays x] nRows x nCols
_SHIFTS = {
    "Right": (-1, 1),
    "Left": (-1, -1),
    "Down": (-2, 1),
    "Up": (-2, -1),
    "Back": (-3, 1),
    "Front": (-3, -1),
}


class _AtomExtension(nn.Module):
    _directions = ()

    def __init__(self, number_of_channels, direction, target_channels, name):
        super().__init__()
        if direction not in self._directions:
            raise InvalidDirection(
                f"{direction} : Direction should be either of {', '.join(self._directions)}"
            )
        if target_channels not in ("Sum", "Difference"):
            raise InvalidTargetChannels(
                f"{target_channels} : TargetChannels should be either of Sum or Difference"
            )
        self.number_of_channels = as_pair(number_of_channels)
        self.direction = direction
        self.target_channels = target_channels
        self.name = name
        ps, pa = self.number_of_channels
        self.description = (
            f"{direction} shift the {target_channels.lower()}-channel Coefs. (ps,pa) = ({ps},{pa})"
        )

    def forward(self, X):
        ps, pa = self.number_of_channels
        dim, shift = _SHIFTS[self.direction]
        Xs = X[:, :ps]
        Xa = X[:, ps : ps + pa]
        Ys = Xs + Xa
        Ya = Xs - Xa
        # Periodic extension as circshift in MATLAB
        if self.target_channels == "Difference":
            Ya = torch.roll(Ya, shifts=shift, dims=dim)
        else:
            Ys = torch.roll(Ys, shifts=shift, dims=dim)
        return 0.5 * torch.cat((Ys + Ya, Ys - Ya), dim=1)


class NsoltAtomExtension2dLayer(_AtomExtension):
    """Atom extension layer, equivalent to MATLAB ``nsoltAtomExtension2dLayer``.

    Input and output: ``nSamples x (ps+pa) x nRows x nCols``.
    ``direction`` is one of 'Right', 'Left', 'Down' and 'Up', and
    ``target_channels`` is 'Sum' or 'Difference'.
    """

    _directions = ("Right", "Left", "Down", "Up")

    def __init__(self, number_of_channels, direction, target_channels, name=""):
        super().__init__(number_of_channels, direction, target_channels, name)


class NsoltAtomExtension3dLayer(_AtomExtension):
    """Atom extension layer, equivalent to MATLAB ``nsoltAtomExtension3dLayer``.

    Input and output: ``nSamples x (ps+pa) x nLays x nRows x nCols``.
    ``direction`` is one of 'Right', 'Left', 'Down', 'Up', 'Back' and 'Front'.
    """

    _directions = ("Right", "Left", "Down", "Up", "Back", "Front")

    def __init__(self, number_of_channels, direction, target_channels, name=""):
        super().__init__(number_of_channels, direction, target_channels, name)
