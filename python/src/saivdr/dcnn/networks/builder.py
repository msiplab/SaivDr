"""Network builders (MATLAB fcn_creatensoltlgraph2d and its deprecated variants)."""

import warnings

from .nsolt import (
    NsoltAnalyzer2d,
    NsoltAnalyzer3d,
    NsoltNetwork2d,
    NsoltNetwork3d,
    NsoltSynthesizer2d,
    NsoltSynthesizer3d,
    _no_dc_leakage_pair,
)

__all__ = ["create_nsolt_2d", "create_nsolt_3d", "creatensoltlgraphs2d", "createsdiplgraph2d"]


def _create(
    classes,
    number_of_components,
    number_of_channels,
    decimation_factor,
    polyphase_order,
    number_of_levels,
    number_of_vanishing_moments,
    mode,
    prefix,
    dtype,
    device,
):
    analyzer_cls, synthesizer_cls, network_cls = classes
    ndl = _no_dc_leakage_pair(number_of_vanishing_moments)
    kw = dict(
        number_of_channels=number_of_channels,
        decimation_factor=decimation_factor,
        polyphase_order=polyphase_order,
        number_of_levels=number_of_levels,
        number_of_components=number_of_components,
        prefix=prefix,
        dtype=dtype,
        device=device,
    )
    if mode == "Whole":
        return network_cls(number_of_vanishing_moments=ndl, **kw)
    if mode == "Analyzer":
        return analyzer_cls(no_dc_leakage=ndl[0], **kw)
    if mode == "Synthesizer":
        return synthesizer_cls(no_dc_leakage=ndl[1], **kw)
    raise ValueError("Mode should be in { 'Whole', 'Analyzer', 'Synthesizer' }")


def create_nsolt_2d(
    number_of_components=1,
    number_of_channels=(2, 2),
    decimation_factor=(2, 2),
    polyphase_order=(0, 0),
    number_of_levels=1,
    number_of_vanishing_moments=(1, 1),
    mode="Whole",
    prefix="",
    dtype=None,
    device=None,
):
    """Create a 2-D NSOLT network, equivalent to MATLAB ``fcn_creatensoltlgraph2d``.

    Parameters
    ----------
    number_of_components : int
        Number of image components (``NumberOfComponents``).
    number_of_channels : (int, int)
        ``(ps, pa)``; only Type-I (``ps == pa``) is supported.
    decimation_factor : (int, int)
        ``(decV, decH)``.
    polyphase_order : (int, int)
        ``(ordV, ordH)``; only even orders are supported.
    number_of_levels : int
        Number of tree levels.
    number_of_vanishing_moments : int or (int, int)
        0 or 1 for (analyzer, synthesizer); 1 means no DC leakage.
    mode : {'Whole', 'Analyzer', 'Synthesizer'}
    prefix : str
        Prefix of the layer names.

    Returns
    -------
    NsoltNetwork2d, NsoltAnalyzer2d or NsoltSynthesizer2d
        All the angles are initialized to zero and all the intermediate
        rotations have ``mus = -1``, so that the network initially
        coincides with the block DCT.
    """
    return _create(
        (NsoltAnalyzer2d, NsoltSynthesizer2d, NsoltNetwork2d),
        number_of_components,
        number_of_channels,
        decimation_factor,
        polyphase_order,
        number_of_levels,
        number_of_vanishing_moments,
        mode,
        prefix,
        dtype,
        device,
    )


def create_nsolt_3d(
    number_of_components=1,
    number_of_channels=(4, 4),
    decimation_factor=(2, 2, 2),
    polyphase_order=(0, 0, 0),
    number_of_levels=1,
    number_of_vanishing_moments=(1, 1),
    mode="Whole",
    prefix="",
    dtype=None,
    device=None,
):
    """Create a 3-D NSOLT network (no MATLAB builder exists; see :mod:`.nsolt`).

    The parameters are those of :func:`create_nsolt_2d` with three elements in
    the MATLAB order (vertical, horizontal, depth).
    """
    return _create(
        (NsoltAnalyzer3d, NsoltSynthesizer3d, NsoltNetwork3d),
        number_of_components,
        number_of_channels,
        decimation_factor,
        polyphase_order,
        number_of_levels,
        number_of_vanishing_moments,
        mode,
        prefix,
        dtype,
        device,
    )


def creatensoltlgraphs2d(
    number_of_channels=(2, 2),
    decimation_factor=(2, 2),
    polyphase_order=(0, 0),
    number_of_levels=1,
    number_of_vanishing_moments=1,
    dtype=None,
    device=None,
):
    """Deprecated: return ``(analyzer, synthesizer)`` (MATLAB ``fcn_creatensoltlgraphs2d``).

    Unlike MATLAB, the layer names include ``Cmp1_`` as in :func:`create_nsolt_2d`.
    """
    warnings.warn("creatensoltlgraphs2d is deprecated; use create_nsolt_2d", DeprecationWarning, stacklevel=2)
    kw = dict(
        number_of_channels=number_of_channels,
        decimation_factor=decimation_factor,
        polyphase_order=polyphase_order,
        number_of_levels=number_of_levels,
        number_of_vanishing_moments=number_of_vanishing_moments,
        dtype=dtype,
        device=device,
    )
    return create_nsolt_2d(mode="Analyzer", **kw), create_nsolt_2d(mode="Synthesizer", **kw)


def createsdiplgraph2d(
    number_of_channels=(2, 2),
    decimation_factor=(2, 2),
    polyphase_order=(0, 0),
    number_of_levels=1,
    number_of_vanishing_moments=(1, 1),
    dtype=None,
    device=None,
):
    """Deprecated: return the whole network (MATLAB ``fcn_createsdiplgraph2d``)."""
    warnings.warn("createsdiplgraph2d is deprecated; use create_nsolt_2d", DeprecationWarning, stacklevel=2)
    return create_nsolt_2d(
        number_of_channels=number_of_channels,
        decimation_factor=decimation_factor,
        polyphase_order=polyphase_order,
        number_of_levels=number_of_levels,
        number_of_vanishing_moments=number_of_vanishing_moments,
        mode="Whole",
        dtype=dtype,
        device=device,
    )
