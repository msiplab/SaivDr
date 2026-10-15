"""Helpers for NSOLT networks (MATLAB fcn_cpparamssyn2ana, fcn_attachserializationlayers)."""

import re

import torch
import torch.nn as nn

from ..layers import (
    NsoltSubbandDeserialization2dLayer,
    NsoltSubbandDeserialization3dLayer,
    NsoltSubbandSerialization2dLayer,
    NsoltSubbandSerialization3dLayer,
)
from ..layers.rotation import _RotationBase

__all__ = [
    "copy_params_syn2ana",
    "copy_params_ana2syn",
    "attach_serialization_layers",
    "NsoltSerializedAnalyzer",
    "NsoltSerializedSynthesizer",
]

_EXP_ANALYZER = re.compile(r"^Lv\d+_Cmp\d+_V(\w\d|0)+$")


def _copy(dst, src):
    with torch.no_grad():
        dst.angles.copy_(src.angles.to(dst.angles))
        dst.mus.copy_(src.mus.to(dst.mus))
    if hasattr(dst, "no_dc_leakage") and hasattr(src, "no_dc_leakage"):
        dst.no_dc_leakage = src.no_dc_leakage


def copy_params_syn2ana(analyzer, synthesizer, verbose=False):
    """Copy ``angles``, ``mus`` (and ``no_dc_leakage``) from layer ``X~`` to ``X``.

    Equivalent to MATLAB ``fcn_cpparamssyn2ana``; ``analyzer`` is modified
    in place and returned.
    """
    for name, alayer in analyzer.layers.items():
        if _EXP_ANALYZER.match(name) and isinstance(alayer, _RotationBase):
            slayer = synthesizer.layers[name + "~"]
            _copy(alayer, slayer)
            if verbose:
                print(f"Copy angles from {name}~ to {name}")
    return analyzer


def copy_params_ana2syn(synthesizer, analyzer, verbose=False):
    """Copy ``angles``, ``mus`` (and ``no_dc_leakage``) from layer ``X`` to ``X~``.

    The counterpart of :func:`copy_params_syn2ana`; ``synthesizer`` is
    modified in place and returned.
    """
    for name, alayer in analyzer.layers.items():
        if _EXP_ANALYZER.match(name) and isinstance(alayer, _RotationBase):
            _copy(synthesizer.layers[name + "~"], alayer)
            if verbose:
                print(f"Copy angles from {name} to {name}~")
    return synthesizer


class NsoltSerializedAnalyzer(nn.Module):
    """Analyzer followed by the subband serialization layer ``Sb_Srz``.

    Output: ``nSamples x nElements``.
    """

    def __init__(self, analyzer, serialization):
        super().__init__()
        self.analyzer = analyzer
        self.serialization = serialization

    def forward(self, X):
        return self.serialization(*self.analyzer(X))


class NsoltSerializedSynthesizer(nn.Module):
    """Subband deserialization layer ``Sb_Dsz`` followed by the synthesizer.

    Input: ``nSamples x nElements``.
    """

    def __init__(self, synthesizer, deserialization):
        super().__init__()
        self.deserialization = deserialization
        self.synthesizer = synthesizer

    def forward(self, X):
        return self.synthesizer(*self.deserialization(X))


def attach_serialization_layers(analyzer, synthesizer, patch_size):
    """Attach the subband (de)serialization layers to the networks.

    Equivalent to MATLAB ``fcn_attachserializationlayers``. ``patch_size``
    is the size of the training patches in the MATLAB order
    ``(height, width[, depth])``. Returns ``(serialized analyzer,
    serialized synthesizer)``. As in MATLAB, only single-component networks
    are supported.
    """
    ndims = len(analyzer.decimation_factor)
    srz_cls, dsz_cls = (
        (NsoltSubbandSerialization2dLayer, NsoltSubbandDeserialization2dLayer)
        if ndims == 2
        else (NsoltSubbandSerialization3dLayer, NsoltSubbandDeserialization3dLayer)
    )
    if analyzer.number_of_components != 1 or synthesizer.number_of_components != 1:
        raise ValueError("Only single-component networks can be serialized")
    srz = srz_cls(
        original_dimension=patch_size,
        number_of_channels=analyzer.number_of_channels,
        decimation_factor=analyzer.decimation_factor,
        number_of_levels=analyzer.number_of_levels,
        name="Sb_Srz",
    )
    dsz = dsz_cls(
        original_dimension=patch_size,
        number_of_channels=synthesizer.number_of_channels,
        decimation_factor=synthesizer.decimation_factor,
        number_of_levels=synthesizer.number_of_levels,
        name="Sb_Dsz",
    )
    return NsoltSerializedAnalyzer(analyzer, srz), NsoltSerializedSynthesizer(synthesizer, dsz)
