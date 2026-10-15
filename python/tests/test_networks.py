"""Properties of the networks that do not need the MATLAB reference data."""

import pytest
import torch

import saivdr.dcnn as dcnn
from saivdr.dcnn import (
    InvalidNumberOfChannels,
    InvalidNumberOfVanishingMoments,
    InvalidPolyPhaseOrder,
    attach_serialization_layers,
    copy_params_syn2ana,
    create_nsolt_2d,
    create_nsolt_3d,
)
from saivdr.dcnn.matlab_compat import rotation_layers

DT = torch.float64


def randomize(net, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.rand(p.shape, generator=g, dtype=p.dtype) * 6.28 - 3.14)
    return net


CONFIGS_2D = [
    dict(number_of_channels=(2, 2), decimation_factor=(2, 2), polyphase_order=(0, 0), number_of_levels=1),
    dict(number_of_channels=(3, 3), decimation_factor=(2, 2), polyphase_order=(2, 2), number_of_levels=2),
    dict(number_of_channels=(5, 5), decimation_factor=(2, 4), polyphase_order=(2, 0), number_of_levels=1),
    dict(
        number_of_channels=(3, 3),
        decimation_factor=(2, 2),
        polyphase_order=(0, 2),
        number_of_levels=2,
        number_of_components=3,
    ),
]


@pytest.mark.parametrize("cfg", CONFIGS_2D)
@pytest.mark.parametrize("vm", [(0, 0), (1, 1)])
def test_whole_2d_perfect_reconstruction(cfg, vm):
    net = create_nsolt_2d(number_of_vanishing_moments=vm, dtype=DT, **cfg)
    randomize(net)
    copy_params_syn2ana(net.analyzer, net.synthesizer)
    x = torch.randn(2, cfg.get("number_of_components", 1), 16, 16, dtype=DT)
    torch.testing.assert_close(net(x), x)


@pytest.mark.parametrize("cfg", CONFIGS_2D)
def test_adjoint(cfg):
    ana = randomize(create_nsolt_2d(mode="Analyzer", dtype=DT, **cfg))
    syn = ana.adjoint()
    x = torch.randn(2, cfg.get("number_of_components", 1), 16, 16, dtype=DT)
    y = ana(x)
    torch.testing.assert_close(syn(*y), x)
    # <A x, y'> == <x, A^T y'> for random coefficients
    yy = tuple(torch.randn_like(t) for t in y)
    lhs = sum((a * b).sum() for a, b in zip(y, yy))
    rhs = (x * syn(*yy)).sum()
    torch.testing.assert_close(lhs, rhs)


def test_zero_angles_coincide_with_block_dct():
    ana = create_nsolt_2d(number_of_channels=(3, 3), polyphase_order=(2, 2), mode="Analyzer", dtype=DT)
    x = torch.randn(1, 1, 8, 8, dtype=DT)
    ac, dc = ana(x)
    y = dcnn.NsoltBlockDct2dLayer((2, 2))(x)
    torch.testing.assert_close(dc, y[:, :1])
    torch.testing.assert_close(ac[:, :1], y[:, 1:2])  # symmetric channels
    torch.testing.assert_close(ac[:, 2:4], y[:, 2:4])  # antisymmetric channels
    assert ac[:, 1].abs().max() == 0 and ac[:, 4].abs().max() == 0


@pytest.mark.parametrize("nlevels", [1, 2])
def test_whole_3d_perfect_reconstruction(nlevels):
    net = create_nsolt_3d(
        number_of_channels=(4, 4), polyphase_order=(2, 0, 2), number_of_levels=nlevels, dtype=DT
    )
    randomize(net)
    copy_params_syn2ana(net.analyzer, net.synthesizer)
    x = torch.randn(2, 1, 8, 8, 8, dtype=DT)
    torch.testing.assert_close(net(x), x)


def test_layer_names_follow_matlab():
    net = create_nsolt_2d(number_of_channels=(3, 3), polyphase_order=(2, 2), number_of_levels=2)
    names = list(net.layers)
    for name in [
        "Lv1_E0",
        "Lv1_Cmp1_V0",
        "Lv1_Cmp1_Qh1rd",
        "Lv1_Cmp1_Vh1",
        "Lv1_Cmp1_Qh2ls",
        "Lv1_Cmp1_Vh2",
        "Lv1_Cmp1_Qv1dd",
        "Lv2_Cmp1_Sp",
        "Lv1_E0~",
        "Lv1_Cmp1_V0~",
        "Lv1_Cmp1_Vh1~",
        "Lv2_Cmp1_Qv2us~",
        "Lv2_Cmp1_Cn",
    ]:
        assert name in names
    assert all(layer.mus.eq(-1).all() for name, layer in rotation_layers(net).items() if "_V0" not in name)


def test_invalid_configurations():
    with pytest.raises(InvalidNumberOfChannels):
        create_nsolt_2d(number_of_channels=(3, 2))
    with pytest.raises(InvalidPolyPhaseOrder):
        create_nsolt_2d(polyphase_order=(1, 0))
    with pytest.raises(InvalidNumberOfVanishingMoments):
        create_nsolt_2d(number_of_vanishing_moments=2)
    with pytest.raises(ValueError):
        create_nsolt_2d(mode="Encoder")


@pytest.mark.parametrize("ndims", [2, 3])
def test_attach_serialization_layers(ndims):
    create = create_nsolt_2d if ndims == 2 else create_nsolt_3d
    nch = (3, 3) if ndims == 2 else (4, 4)
    ana = randomize(create(number_of_channels=nch, number_of_levels=2, mode="Analyzer", dtype=DT))
    syn = ana.adjoint()
    size = (16,) * ndims
    sana, ssyn = attach_serialization_layers(ana, syn, size)
    x = torch.randn((2, 1) + size, dtype=DT)
    z = sana(x)
    assert z.shape == (2, sana.serialization.number_of_elements)
    torch.testing.assert_close(ssyn(z), x)


def test_atomic_images():
    syn = create_nsolt_2d(
        number_of_channels=(3, 3), polyphase_order=(2, 2), number_of_levels=2, mode="Synthesizer", dtype=DT
    )
    randomize(syn)
    atoms = dcnn.atomic_images(syn, (16, 16))
    assert atoms.shape == (11, 1, 16, 16)
    # Each atom is the response to a unit impulse at index round(end/2) of a subband
    coefs = [
        torch.zeros(1, 5, 8, 8, dtype=DT),
        torch.zeros(1, 5, 4, 4, dtype=DT),
        torch.zeros(1, 1, 4, 4, dtype=DT),
    ]
    coefs[0][0, 2, 3, 3] = 1
    torch.testing.assert_close(atoms[8], syn(*coefs)[0])
    assert dcnn.visualize.default_patch_size(syn) == (24, 24)


def test_deprecated_builders():
    with pytest.warns(DeprecationWarning):
        ana, syn = dcnn.creatensoltlgraphs2d()
    assert isinstance(ana, dcnn.NsoltAnalyzer2d) and isinstance(syn, dcnn.NsoltSynthesizer2d)
    with pytest.warns(DeprecationWarning):
        assert isinstance(dcnn.createsdiplgraph2d(), dcnn.NsoltNetwork2d)
