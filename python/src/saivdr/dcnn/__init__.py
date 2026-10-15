"""NSOLT layers and networks for PyTorch, ported from MATLAB ``+saivdr/+dcnn``.

Arrays follow the PyTorch convention: images are ``N x C x H x W`` (2-D) or
``N x C x D x H x W`` (3-D), and transform coefficients are
``N x nChs x nRows x nCols`` (2-D) or ``N x nChs x nLays x nRows x nCols``
(3-D). Parameters such as ``decimation_factor`` keep the MATLAB order
(vertical, horizontal[, depth]). See :mod:`saivdr.dcnn.matlab_compat` for the
conversion from and to MATLAB.
"""

import importlib
import warnings

from . import matlab_compat
from ._dct import block_dct_matrix_2d, block_dct_matrix_3d, dctmtx
from ._orthmtx import orthmtxgen
from .exceptions import (
    InvalidDirection,
    InvalidMode,
    InvalidMus,
    InvalidNumberOfAngles,
    InvalidNumberOfChannels,
    InvalidNumberOfLevels,
    InvalidNumberOfVanishingMoments,
    InvalidPolyPhaseOrder,
    InvalidTargetChannels,
    NsoltLayerError,
)
from .layers import *  # noqa: F401,F403
from .layers import __all__ as _layers_all
from .networks import *  # noqa: F401,F403
from .networks import __all__ as _networks_all
from .utility import ChannelGroup, Direction
from .visualize import atomic_images, atomic_imshow

__all__ = [
    *_layers_all,
    *_networks_all,
    "ChannelGroup",
    "Direction",
    "InvalidDirection",
    "InvalidMode",
    "InvalidMus",
    "InvalidNumberOfAngles",
    "InvalidNumberOfChannels",
    "InvalidNumberOfLevels",
    "InvalidNumberOfVanishingMoments",
    "InvalidPolyPhaseOrder",
    "InvalidTargetChannels",
    "NsoltLayerError",
    "atomic_images",
    "atomic_imshow",
    "block_dct_matrix_2d",
    "block_dct_matrix_3d",
    "dctmtx",
    "matlab_compat",
    "orthmtxgen",
]

# Names of the former appendix/torch_nsolt API, resolved lazily from
# saivdr.dcnn.legacy with a DeprecationWarning
_LEGACY = {
    "NsoltAnalysis2dNetwork": "analysis_2d",
    "NsoltSynthesis2dNetwork": "synthesis_2d",
    "OrthonormalTransform": "orthonormal_transform",
    "OrthonormalMatrixGenerationSystem": "utility",
}


def __getattr__(name):
    if name in _LEGACY:
        warnings.warn(
            f"saivdr.dcnn.{name} is the deprecated channel-last implementation in "
            "saivdr.dcnn.legacy; use create_nsolt_2d or the new layers instead",
            DeprecationWarning,
            stacklevel=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            module = importlib.import_module(f".legacy.{_LEGACY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
