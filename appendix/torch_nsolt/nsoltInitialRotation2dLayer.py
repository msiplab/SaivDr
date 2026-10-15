"""Deprecated: moved to saivdr.dcnn.legacy.initial_rotation_2d (install with pip install ./python)."""

import warnings

warnings.warn(
    "appendix/torch_nsolt is deprecated; use saivdr.dcnn (pip install ./python)",
    DeprecationWarning,
    stacklevel=2,
)

from saivdr.dcnn.legacy.initial_rotation_2d import *  # noqa: E402,F401,F403
