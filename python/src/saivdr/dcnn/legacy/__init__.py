"""Legacy PyTorch NSOLT modules (formerly ``appendix/torch_nsolt``).

These modules keep the original channel-last coefficient layout
(nSamples x nRows x nCols x nChs) and are kept only for backward
compatibility. Use :mod:`saivdr.dcnn` instead.
"""

import warnings

warnings.warn(
    "saivdr.dcnn.legacy is deprecated and will be removed in a future release; "
    "use saivdr.dcnn instead",
    DeprecationWarning,
    stacklevel=2,
)
