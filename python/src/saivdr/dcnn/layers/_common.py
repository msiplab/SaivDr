"""Helpers shared by the NSOLT layers.

Coefficient arrays are laid out as ``nSamples x nChs x <spatial>``, where
``<spatial>`` is ``nRows x nCols`` in 2-D and ``nLays x nRows x nCols`` in
3-D. MATLAB uses ``nChs x nRows x nCols (x nLays) x nSamples`` instead.
"""

import torch

__all__ = ["apply_channel_matrix", "as_pair"]


def apply_channel_matrix(M, X):
    """Multiply ``M`` from the left along the channel dimension of ``X``.

    ``X`` has the shape ``nSamples x nCols(M) x <spatial>`` and the result has
    the shape ``nSamples x nRows(M) x <spatial>``.
    """
    Y = torch.movedim(X, 1, -1) @ M.to(dtype=X.dtype).T
    return torch.movedim(Y, -1, 1)


def as_pair(number_of_channels):
    """Return ``(ps, pa)`` as Python ints."""
    ps, pa = (int(v) for v in number_of_channels)
    return ps, pa
