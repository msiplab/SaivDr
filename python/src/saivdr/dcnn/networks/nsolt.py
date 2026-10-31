"""NSOLT analyzer/synthesizer networks.

Equivalent to the layer graphs built by MATLAB
``saivdr.dcnn.fcn_creatensoltlgraph2d``. The layers are kept in an
:class:`torch.nn.ModuleDict` whose keys are the MATLAB layer names (e.g.
``'Lv1_Cmp1_V0'`` and ``'Lv1_Cmp1_Vh1~'``), so that parameters can be
exchanged with MATLAB by name. The 3-D networks follow the same
construction with the 3-D layers; the atoms are extended in the depth,
horizontal and vertical directions in this order, as in
``saivdr.dictionary.nsoltx.NsoltAtomExtender3d``.
"""

import torch
import torch.nn as nn

from ..exceptions import (
    InvalidNumberOfChannels,
    InvalidNumberOfLevels,
    InvalidNumberOfVanishingMoments,
    InvalidPolyPhaseOrder,
)
from ..layers import (
    NsoltAtomExtension2dLayer,
    NsoltAtomExtension3dLayer,
    NsoltBlockDct2dLayer,
    NsoltBlockDct3dLayer,
    NsoltBlockIdct2dLayer,
    NsoltBlockIdct3dLayer,
    NsoltChannelConcatenation2dLayer,
    NsoltChannelConcatenation3dLayer,
    NsoltChannelSeparation2dLayer,
    NsoltChannelSeparation3dLayer,
    NsoltComponentSeparation2dLayer,
    NsoltFinalRotation2dLayer,
    NsoltFinalRotation3dLayer,
    NsoltInitialRotation2dLayer,
    NsoltInitialRotation3dLayer,
    NsoltIntermediateRotation2dLayer,
    NsoltIntermediateRotation3dLayer,
)
from ..utility import Direction

__all__ = [
    "NsoltAnalyzer2d",
    "NsoltSynthesizer2d",
    "NsoltNetwork2d",
    "NsoltAnalyzer3d",
    "NsoltSynthesizer3d",
    "NsoltNetwork3d",
]


class _Spec:
    """Layer classes and extension steps of each dimensionality."""

    def __init__(
        self, ndims, dct, idct, initial, final, intermediate, extension, separation, concatenation, steps
    ):
        self.ndims = ndims
        self.dct = dct
        self.idct = idct
        self.initial = initial
        self.final = final
        self.intermediate = intermediate
        self.extension = extension
        self.separation = separation
        self.concatenation = concatenation
        # (polyphase-order index, label, (analysis directions), (synthesis directions),
        #  (analysis suffixes))
        self.steps = steps


_SPEC2D = _Spec(
    2,
    NsoltBlockDct2dLayer,
    NsoltBlockIdct2dLayer,
    NsoltInitialRotation2dLayer,
    NsoltFinalRotation2dLayer,
    NsoltIntermediateRotation2dLayer,
    NsoltAtomExtension2dLayer,
    NsoltChannelSeparation2dLayer,
    NsoltChannelConcatenation2dLayer,
    (
        (Direction.HORIZONTAL, "h", ("Right", "Left"), ("Left", "Right"), ("rd", "ls")),
        (Direction.VERTICAL, "v", ("Down", "Up"), ("Up", "Down"), ("dd", "us")),
    ),
)

_SPEC3D = _Spec(
    3,
    NsoltBlockDct3dLayer,
    NsoltBlockIdct3dLayer,
    NsoltInitialRotation3dLayer,
    NsoltFinalRotation3dLayer,
    NsoltIntermediateRotation3dLayer,
    NsoltAtomExtension3dLayer,
    NsoltChannelSeparation3dLayer,
    NsoltChannelConcatenation3dLayer,
    (
        (Direction.DEPTH, "d", ("Back", "Front"), ("Front", "Back"), ("bd", "fs")),
        (Direction.HORIZONTAL, "h", ("Right", "Left"), ("Left", "Right"), ("rd", "ls")),
        (Direction.VERTICAL, "v", ("Down", "Up"), ("Up", "Down"), ("dd", "us")),
    ),
)


def _no_dc_leakage_pair(number_of_vanishing_moments):
    v = number_of_vanishing_moments
    if isinstance(v, (bool, int, float)):
        v = (v, v)
    v = tuple(int(bool(x)) if isinstance(x, bool) else int(x) for x in v)
    if len(v) != 2 or any(x not in (0, 1) for x in v):
        raise InvalidNumberOfVanishingMoments(
            f"{number_of_vanishing_moments} : The number of vanishing moments should be 0 or 1"
        )
    return v


class _NsoltConfig:
    def __init__(
        self,
        spec,
        number_of_channels,
        decimation_factor,
        polyphase_order,
        number_of_levels,
        number_of_components,
        prefix,
    ):
        ps, pa = (int(v) for v in number_of_channels)
        if ps != pa:
            raise InvalidNumberOfChannels(
                f"[{ps} {pa}] : Currently, Type-I NSOLT is only suported, where the even and odd "
                "channel numbers should be the same."
            )
        polyphase_order = tuple(int(v) for v in polyphase_order)
        if len(polyphase_order) != spec.ndims or any(v % 2 for v in polyphase_order):
            raise InvalidPolyPhaseOrder(
                f"{polyphase_order} : Currently, even polyphase orders are only supported."
            )
        decimation_factor = tuple(int(v) for v in decimation_factor)
        if len(decimation_factor) != spec.ndims:
            raise ValueError(f"decimation_factor should have {spec.ndims} elements")
        if int(number_of_levels) < 1:
            raise InvalidNumberOfLevels(f"{number_of_levels} : The number of levels should be positive")
        self.spec = spec
        self.number_of_channels = (ps, pa)
        self.decimation_factor = decimation_factor
        self.polyphase_order = polyphase_order
        self.number_of_levels = int(number_of_levels)
        self.number_of_components = int(number_of_components)
        self.prefix = prefix

    def kwargs(self):
        return dict(
            number_of_channels=self.number_of_channels,
            decimation_factor=self.decimation_factor,
            polyphase_order=self.polyphase_order,
            number_of_levels=self.number_of_levels,
            number_of_components=self.number_of_components,
            prefix=self.prefix,
        )


class _NsoltBase(nn.Module):
    _spec = None

    def __init__(
        self,
        number_of_channels,
        decimation_factor,
        polyphase_order,
        number_of_levels,
        number_of_components,
        prefix,
    ):
        super().__init__()
        self.config = _NsoltConfig(
            self._spec,
            number_of_channels,
            decimation_factor,
            polyphase_order,
            number_of_levels,
            number_of_components,
            prefix,
        )
        self.layers = nn.ModuleDict()

    # Properties mirroring the MATLAB name-value arguments
    @property
    def number_of_channels(self):
        return self.config.number_of_channels

    @property
    def decimation_factor(self):
        return self.config.decimation_factor

    @property
    def polyphase_order(self):
        return self.config.polyphase_order

    @property
    def number_of_levels(self):
        return self.config.number_of_levels

    @property
    def number_of_components(self):
        return self.config.number_of_components

    @property
    def prefix(self):
        return self.config.prefix

    def _add(self, layer):
        self.layers[layer.name] = layer
        return layer.name

    def _names(self, iLv, iCmp=None):
        strLv = f"{self.prefix}Lv{iLv}_"
        return strLv if iCmp is None else f"{strLv}Cmp{iCmp}_"


class _NsoltAnalyzer(_NsoltBase):
    """Analyzer part of the network (``Mode='Analyzer'`` in MATLAB)."""

    def __init__(
        self,
        number_of_channels=(2, 2),
        decimation_factor=None,
        polyphase_order=None,
        number_of_levels=1,
        no_dc_leakage=True,
        number_of_components=1,
        prefix="",
        dtype=None,
        device=None,
    ):
        nd = self._spec.ndims
        decimation_factor = (2,) * nd if decimation_factor is None else decimation_factor
        polyphase_order = (0,) * nd if polyphase_order is None else polyphase_order
        super().__init__(
            number_of_channels,
            decimation_factor,
            polyphase_order,
            number_of_levels,
            number_of_components,
            prefix,
        )
        self.no_dc_leakage = bool(no_dc_leakage)
        spec = self._spec
        cfg = self.config
        nchs = cfg.number_of_channels
        self._dct_names = []
        self._chains = []  # [iLv][iCmp] -> list of layer names
        for iLv in range(1, cfg.number_of_levels + 1):
            strLv = self._names(iLv)
            self._dct_names.append(
                self._add(
                    spec.dct(
                        decimation_factor=cfg.decimation_factor,
                        number_of_components=cfg.number_of_components,
                        name=strLv + "E0",
                    )
                )
            )
            chains = []
            for iCmp in range(1, cfg.number_of_components + 1):
                s = self._names(iLv, iCmp)
                chain = [
                    self._add(
                        spec.initial(
                            nchs,
                            cfg.decimation_factor,
                            no_dc_leakage=self.no_dc_leakage,
                            name=s + "V0",
                            dtype=dtype,
                            device=device,
                        )
                    )
                ]
                for idx, label, dirs, _, sfx in spec.steps:
                    for iOrder in range(2, cfg.polyphase_order[idx] + 1, 2):
                        chain += [
                            self._add(
                                spec.extension(
                                    nchs, dirs[0], "Difference", name=f"{s}Q{label}{iOrder - 1}{sfx[0]}"
                                )
                            ),
                            self._add(
                                spec.intermediate(
                                    nchs,
                                    mode="Analysis",
                                    mus=-1,
                                    name=f"{s}V{label}{iOrder - 1}",
                                    dtype=dtype,
                                    device=device,
                                )
                            ),
                            self._add(
                                spec.extension(nchs, dirs[1], "Sum", name=f"{s}Q{label}{iOrder}{sfx[1]}")
                            ),
                            self._add(
                                spec.intermediate(
                                    nchs,
                                    mode="Analysis",
                                    mus=-1,
                                    name=f"{s}V{label}{iOrder}",
                                    dtype=dtype,
                                    device=device,
                                )
                            ),
                        ]
                chain.append(self._add(spec.separation(name=s + "Sp")))
                chains.append(chain)
            self._chains.append(chains)

    def forward(self, X):
        """Analyze ``X``.

        Returns the tuple ``(Lv1_AcOut, ..., LvN_AcOut, LvN_DcOut)`` in the
        order of the MATLAB network outputs. ``LvK_AcOut`` has
        ``nComponents*(nChs-1)`` channels and ``LvN_DcOut`` has ``nComponents``.
        """
        outputs = []
        dc = X
        for dct_name, chains in zip(self._dct_names, self._chains):
            comps = self.layers[dct_name](dc)
            if not isinstance(comps, tuple):
                comps = (comps,)
            acs, dcs = [], []
            for Y, chain in zip(comps, chains):
                for name in chain[:-1]:
                    Y = self.layers[name](Y)
                ac, dc_ = self.layers[chain[-1]](Y)
                acs.append(ac)
                dcs.append(dc_)
            outputs.append(torch.cat(acs, dim=1))
            dc = torch.cat(dcs, dim=1)
        outputs.append(dc)
        return tuple(outputs)

    def adjoint(self):
        """Return the synthesizer whose parameters are copied from this analyzer.

        With the same parameters, the synthesizer is the adjoint (inverse)
        of the analyzer.
        """
        syn = self._synthesizer_class(**self.config.kwargs(), no_dc_leakage=self.no_dc_leakage)
        syn.to(device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)
        from .utils import copy_params_ana2syn

        copy_params_ana2syn(syn, self)
        return syn

    @property
    def T(self):
        """Alias of :meth:`adjoint`."""
        return self.adjoint()


class _NsoltSynthesizer(_NsoltBase):
    """Synthesizer part of the network (``Mode='Synthesizer'`` in MATLAB)."""

    def __init__(
        self,
        number_of_channels=(2, 2),
        decimation_factor=None,
        polyphase_order=None,
        number_of_levels=1,
        no_dc_leakage=True,
        number_of_components=1,
        prefix="",
        dtype=None,
        device=None,
    ):
        nd = self._spec.ndims
        decimation_factor = (2,) * nd if decimation_factor is None else decimation_factor
        polyphase_order = (0,) * nd if polyphase_order is None else polyphase_order
        super().__init__(
            number_of_channels,
            decimation_factor,
            polyphase_order,
            number_of_levels,
            number_of_components,
            prefix,
        )
        self.no_dc_leakage = bool(no_dc_leakage)
        spec = self._spec
        cfg = self.config
        nchs = cfg.number_of_channels
        nComponents = cfg.number_of_components
        self._idct_names = []
        self._chains = []  # [iLv][iCmp] -> list of layer names in the order of application
        self._acin = []
        self._dcin = []
        for iLv in range(1, cfg.number_of_levels + 1):
            strLv = self._names(iLv)
            self._idct_names.append(
                self._add(
                    spec.idct(
                        decimation_factor=cfg.decimation_factor,
                        number_of_components=nComponents,
                        name=strLv + "E0~",
                    )
                )
            )
            self._acin.append(self._add(NsoltComponentSeparation2dLayer(nComponents, name=strLv + "AcIn")))
            self._dcin.append(self._add(NsoltComponentSeparation2dLayer(nComponents, name=strLv + "DcIn")))
            chains = []
            for iCmp in range(1, nComponents + 1):
                s = self._names(iLv, iCmp)
                chain = [
                    self._add(
                        spec.final(
                            nchs,
                            cfg.decimation_factor,
                            no_dc_leakage=self.no_dc_leakage,
                            name=s + "V0~",
                            dtype=dtype,
                            device=device,
                        )
                    )
                ]
                for idx, label, _, dirs, sfx in spec.steps:
                    for iOrder in range(2, cfg.polyphase_order[idx] + 1, 2):
                        chain += [
                            self._add(
                                spec.extension(
                                    nchs, dirs[0], "Difference", name=f"{s}Q{label}{iOrder - 1}{sfx[0]}~"
                                )
                            ),
                            self._add(
                                spec.intermediate(
                                    nchs,
                                    mode="Synthesis",
                                    mus=-1,
                                    name=f"{s}V{label}{iOrder - 1}~",
                                    dtype=dtype,
                                    device=device,
                                )
                            ),
                            self._add(
                                spec.extension(nchs, dirs[1], "Sum", name=f"{s}Q{label}{iOrder}{sfx[1]}~")
                            ),
                            self._add(
                                spec.intermediate(
                                    nchs,
                                    mode="Synthesis",
                                    mus=-1,
                                    name=f"{s}V{label}{iOrder}~",
                                    dtype=dtype,
                                    device=device,
                                )
                            ),
                        ]
                chain.append(self._add(spec.concatenation(name=s + "Cn")))
                chains.append(chain[::-1])  # applied from Cn to V0~
            self._chains.append(chains)

    def forward(self, *inputs):
        """Synthesize from ``(Lv1_Ac, ..., LvN_Ac, LvN_Dc)``.

        The inputs are in the order of the MATLAB network inputs; a single
        tuple or list is also accepted.
        """
        if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
            inputs = tuple(inputs[0])
        nLevels = self.number_of_levels
        if len(inputs) != nLevels + 1:
            raise ValueError(f"{nLevels + 1} inputs are expected, but {len(inputs)} were given")
        dc = inputs[nLevels]
        for iLv in range(nLevels, 0, -1):
            i = iLv - 1
            acs = self.layers[self._acin[i]](inputs[i])
            dcs = self.layers[self._dcin[i]](dc)
            if not isinstance(acs, tuple):
                acs, dcs = (acs,), (dcs,)
            outs = []
            for ac, dc_, chain in zip(acs, dcs, self._chains[i]):
                Y = self.layers[chain[0]](ac, dc_)
                for name in chain[1:]:
                    Y = self.layers[name](Y)
                outs.append(Y)
            dc = self.layers[self._idct_names[i]](*outs)
        return dc

    def adjoint(self):
        """Return the analyzer whose parameters are copied from this synthesizer."""
        ana = self._analyzer_class(**self.config.kwargs(), no_dc_leakage=self.no_dc_leakage)
        ana.to(device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)
        from .utils import copy_params_syn2ana

        copy_params_syn2ana(ana, self)
        return ana

    @property
    def T(self):
        """Alias of :meth:`adjoint`."""
        return self.adjoint()


class _NsoltNetwork(nn.Module):
    """Analyzer followed by synthesizer (``Mode='Whole'`` in MATLAB)."""

    def __init__(
        self,
        number_of_channels=(2, 2),
        decimation_factor=None,
        polyphase_order=None,
        number_of_levels=1,
        number_of_vanishing_moments=(1, 1),
        number_of_components=1,
        prefix="",
        dtype=None,
        device=None,
    ):
        super().__init__()
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
        self.analyzer = self._analyzer_class(no_dc_leakage=ndl[0], **kw)
        self.synthesizer = self._synthesizer_class(no_dc_leakage=ndl[1], **kw)

    @property
    def layers(self):
        """Dictionary of all the layers keyed by the MATLAB layer names."""
        return {**self.analyzer.layers, **self.synthesizer.layers}

    def forward(self, X):
        return self.synthesizer(*self.analyzer(X))


class NsoltAnalyzer2d(_NsoltAnalyzer):
    """2-D NSOLT analyzer.

    Input: ``nSamples x nComponents x height x width``.
    See :func:`saivdr.dcnn.create_nsolt_2d` for the parameters.
    """

    _spec = _SPEC2D


class NsoltSynthesizer2d(_NsoltSynthesizer):
    """2-D NSOLT synthesizer. Output: ``nSamples x nComponents x height x width``."""

    _spec = _SPEC2D


class NsoltNetwork2d(_NsoltNetwork):
    """2-D NSOLT analyzer followed by the synthesizer."""


class NsoltAnalyzer3d(_NsoltAnalyzer):
    """3-D NSOLT analyzer.

    Input: ``nSamples x nComponents x depth x height x width``. Parameters
    such as ``decimation_factor`` are in the MATLAB order (vertical,
    horizontal, depth).
    """

    _spec = _SPEC3D


class NsoltSynthesizer3d(_NsoltSynthesizer):
    """3-D NSOLT synthesizer. Output: ``nSamples x nComponents x depth x height x width``."""

    _spec = _SPEC3D


class NsoltNetwork3d(_NsoltNetwork):
    """3-D NSOLT analyzer followed by the synthesizer."""


for _ana, _syn, _net in (
    (NsoltAnalyzer2d, NsoltSynthesizer2d, NsoltNetwork2d),
    (NsoltAnalyzer3d, NsoltSynthesizer3d, NsoltNetwork3d),
):
    _ana._synthesizer_class = _syn
    _syn._analyzer_class = _ana
    _net._analyzer_class = _ana
    _net._synthesizer_class = _syn
del _ana, _syn, _net
