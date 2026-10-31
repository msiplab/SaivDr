"""Exceptions raised by NSOLT layers (MATLAB ``NsoltLayer:*`` identifiers)."""

__all__ = [
    "NsoltLayerError",
    "InvalidDirection",
    "InvalidTargetChannels",
    "InvalidMode",
    "InvalidMus",
    "InvalidNumberOfChannels",
    "InvalidPolyPhaseOrder",
    "InvalidNumberOfVanishingMoments",
    "InvalidNumberOfLevels",
    "InvalidNumberOfAngles",
]


class NsoltLayerError(ValueError):
    """Base class of the exceptions raised by NSOLT layers."""


class InvalidDirection(NsoltLayerError):
    """NsoltLayer:InvalidDirection"""


class InvalidTargetChannels(NsoltLayerError):
    """NsoltLayer:InvalidTargetChannels"""


class InvalidMode(NsoltLayerError):
    """NsoltLayer:InvalidMode"""


class InvalidMus(NsoltLayerError):
    """Elements of mus are not +1 or -1."""


class InvalidNumberOfChannels(NsoltLayerError):
    """NsoltLayer:InvalidNumberOfChannels"""


class InvalidPolyPhaseOrder(NsoltLayerError):
    """NsoltLayer:InvalidPolyPhaseOrder"""


class InvalidNumberOfVanishingMoments(NsoltLayerError):
    """NsoltLayer:InvalidNumberOfVanishingMoments"""


class InvalidNumberOfLevels(NsoltLayerError):
    """NsoltLayer:InvalidNumberOfLevels"""


class InvalidNumberOfAngles(NsoltLayerError):
    """The number of angles does not match the number of channels."""
