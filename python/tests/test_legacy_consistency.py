"""The new layers agree with the former appendix/torch_nsolt implementation.

The legacy layers use the channel-last layout nSamples x nRows x nCols x nChs.
"""

import warnings

import pytest
import torch

import saivdr.dcnn as dcnn

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from saivdr.dcnn.legacy import (
        atom_extension_2d,
        block_dct_2d,
        block_dct_3d,
        initial_rotation_2d,
        intermediate_rotation_2d,
    )

DT = torch.float64


def last(t):  # N x C x ... -> N x ... x C
    return torch.movedim(t, 1, -1)


@pytest.mark.parametrize("dec", [(2, 2), (2, 4), (4, 1)])
def test_block_dct_2d(dec):
    x = torch.randn(2, 1, 8, 8, dtype=DT)
    old = block_dct_2d.NsoltBlockDct2dLayer(decimation_factor=list(dec))(x)
    torch.testing.assert_close(last(dcnn.NsoltBlockDct2dLayer(dec)(x)), old)


@pytest.mark.parametrize("dec", [(2, 2, 2), (1, 2, 4), (4, 4, 2)])
def test_block_dct_3d(dec):
    x = torch.randn(2, 1, 8, 8, 8, dtype=DT)  # legacy: N x C x H x W x D
    old = block_dct_3d.NsoltBlockDct3dLayer(decimation_factor=list(dec))(x)  # N x R x C x L x nDecs
    new = dcnn.NsoltBlockDct3dLayer(dec)(x.permute(0, 1, 4, 2, 3))  # N x nDecs x L x R x C
    torch.testing.assert_close(new.permute(0, 3, 4, 2, 1), old)


@pytest.mark.parametrize("ndl", [False, True])
def test_initial_rotation(ndl):
    torch.manual_seed(0)
    old = initial_rotation_2d.NsoltInitialRotation2dLayer(
        number_of_channels=[4, 4], decimation_factor=[2, 2], no_dc_leakage=ndl
    ).double()
    wa, ua = torch.randn(6, dtype=DT), torch.randn(6, dtype=DT)
    with torch.no_grad():
        old.orthTransW0.angles.copy_(wa)
        old.orthTransU0.angles.copy_(ua)
    old.orthTransW0.mus = torch.tensor([-1.0, 1, 1, -1], dtype=DT)
    new = dcnn.NsoltInitialRotation2dLayer((4, 4), (2, 2), no_dc_leakage=ndl, dtype=DT)
    new.set_angles(torch.cat((wa, ua)))
    new.set_mus([-1, 1, 1, -1, 1, 1, 1, 1])
    x = torch.randn(2, 4, 3, 3, dtype=DT)
    torch.testing.assert_close(last(new(x)), old(last(x).contiguous()))


@pytest.mark.parametrize("mode", ["Analysis", "Synthesis"])
def test_intermediate_rotation(mode):
    old = intermediate_rotation_2d.NsoltIntermediateRotation2dLayer(
        number_of_channels=[4, 4], mode=mode, mus=-1
    ).double()
    angles = torch.randn(6, dtype=DT)
    with torch.no_grad():
        old.orthTransUn.angles.copy_(angles)
    new = dcnn.NsoltIntermediateRotation2dLayer((4, 4), mode, mus=-1, angles=angles, dtype=DT)
    x = torch.randn(2, 8, 3, 3, dtype=DT)
    torch.testing.assert_close(last(new(x)), old(last(x).contiguous()))


@pytest.mark.parametrize("direction", ["Right", "Left", "Down", "Up"])
@pytest.mark.parametrize("target", ["Sum", "Difference"])
def test_atom_extension(direction, target):
    old = atom_extension_2d.NsoltAtomExtension2dLayer(
        number_of_channels=[3, 3], direction=direction, target_channels=target
    )
    new = dcnn.NsoltAtomExtension2dLayer((3, 3), direction, target)
    x = torch.randn(2, 6, 4, 5, dtype=DT)
    torch.testing.assert_close(last(new(x)), old(last(x).contiguous()))
