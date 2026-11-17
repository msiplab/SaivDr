"""Channel separation and concatenation layers.

MATLAB nsoltChannelSeparation2d/3dLayer, nsoltChannelConcatenation2d/3dLayer
and nsoltComponentSeparation2dLayer.
"""

import torch
import torch.nn as nn

__all__ = [
    "NsoltChannelSeparation2dLayer",
    "NsoltChannelSeparation3dLayer",
    "NsoltChannelConcatenation2dLayer",
    "NsoltChannelConcatenation3dLayer",
    "NsoltComponentSeparation2dLayer",
]


class _ChannelSeparation(nn.Module):
    def __init__(self, name=""):
        super().__init__()
        self.name = name
        self.description = "Channel separation"

    def forward(self, X):
        """Return ``(ac, dc)``, where ``dc`` is channel 0 and ``ac`` the others."""
        return X[:, 1:], X[:, :1]


class NsoltChannelSeparation2dLayer(_ChannelSeparation):
    """Equivalent to MATLAB ``nsoltChannelSeparation2dLayer``.

    Input: ``nSamples x nChs x nRows x nCols``.
    Outputs: ``ac`` (``nSamples x (nChs-1) x nRows x nCols``) and
    ``dc`` (``nSamples x 1 x nRows x nCols``), in this order.
    """


class NsoltChannelSeparation3dLayer(_ChannelSeparation):
    """Equivalent to MATLAB ``nsoltChannelSeparation3dLayer``."""


class _ChannelConcatenation(nn.Module):
    def __init__(self, name=""):
        super().__init__()
        self.name = name
        self.description = "Channel concatenation"

    def forward(self, ac, dc):
        """Concatenate ``dc`` and ``ac`` in this order along the channels."""
        return torch.cat((dc, ac), dim=1)


class NsoltChannelConcatenation2dLayer(_ChannelConcatenation):
    """Equivalent to MATLAB ``nsoltChannelConcatenation2dLayer`` (inputs ``ac``, ``dc``)."""


class NsoltChannelConcatenation3dLayer(_ChannelConcatenation):
    """Equivalent to MATLAB ``nsoltChannelConcatenation3dLayer`` (inputs ``ac``, ``dc``)."""


class NsoltComponentSeparation2dLayer(nn.Module):
    """Equivalent to MATLAB ``nsoltComponentSeparation2dLayer``.

    Splits the channels into ``number_of_components`` groups of the same
    size. A tensor is returned for one component and a tuple otherwise.
    It works for 3-D arrays as well.
    """

    def __init__(self, number_of_components, name=""):
        super().__init__()
        if number_of_components < 1:
            raise ValueError("numComponents should be greater than or equal to one.")
        self.number_of_components = int(number_of_components)
        self.name = name

    def forward(self, X):
        if self.number_of_components == 1:
            return X
        return tuple(torch.chunk(X, self.number_of_components, dim=1))
