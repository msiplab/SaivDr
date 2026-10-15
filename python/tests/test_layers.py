"""Properties of the layers that do not need the MATLAB reference data."""

import pytest
import torch

import saivdr.dcnn as dcnn
from saivdr.dcnn import (
    InvalidDirection,
    InvalidMode,
    InvalidMus,
    InvalidNumberOfAngles,
    InvalidTargetChannels,
)

DT = torch.float64


@pytest.mark.parametrize("dec", [(1, 1), (2, 2), (2, 4), (4, 1)])
@pytest.mark.parametrize("ncmp", [1, 3])
def test_block_dct_2d_round_trip(dec, ncmp):
    x = torch.randn(2, ncmp, 8, 8, dtype=DT)
    dct = dcnn.NsoltBlockDct2dLayer(dec, ncmp)
    idct = dcnn.NsoltBlockIdct2dLayer(dec, ncmp)
    y = dct(x)
    ys = y if isinstance(y, tuple) else (y,)
    assert all(t.shape == (2, dec[0] * dec[1], 8 // dec[0], 8 // dec[1]) for t in ys)
    torch.testing.assert_close(idct(*ys), x)


@pytest.mark.parametrize("dec", [(1, 1, 1), (2, 2, 2), (1, 2, 4), (4, 4, 2)])
def test_block_dct_3d_round_trip(dec):
    x = torch.randn(2, 1, 8, 8, 8, dtype=DT)
    y = dcnn.NsoltBlockDct3dLayer(dec)(x)
    assert y.shape == (2, dec[0] * dec[1] * dec[2], 8 // dec[2], 8 // dec[0], 8 // dec[1])
    torch.testing.assert_close(dcnn.NsoltBlockIdct3dLayer(dec)(y), x)


def test_block_dct_dc_coefficient():
    x = torch.ones(1, 1, 4, 4, dtype=DT)
    y = dcnn.NsoltBlockDct2dLayer((2, 2))(x)
    torch.testing.assert_close(y[:, 0], torch.full((1, 2, 2), 2.0, dtype=DT))
    torch.testing.assert_close(y[:, 1:], torch.zeros(1, 3, 2, 2, dtype=DT))


def test_block_dct_indivisible():
    with pytest.raises(ValueError):
        dcnn.NsoltBlockDct2dLayer((2, 2))(torch.zeros(1, 1, 5, 4))


@pytest.mark.parametrize("ndl", [False, True])
def test_initial_final_rotation_are_adjoint(ndl):
    torch.manual_seed(0)
    ini = dcnn.NsoltInitialRotation2dLayer((4, 4), (2, 2), no_dc_leakage=ndl, dtype=DT)
    fin = dcnn.NsoltFinalRotation2dLayer((4, 4), (2, 2), no_dc_leakage=ndl, dtype=DT)
    angles = torch.randn(12, dtype=DT)
    for layer in (ini, fin):
        layer.set_angles(angles)
        layer.set_mus([1, -1, 1, 1, -1, 1, -1, 1])
    x = torch.randn(2, 4, 3, 3, dtype=DT)
    torch.testing.assert_close(fin(ini(x)), x)


def test_no_dc_leakage():
    layer = dcnn.NsoltInitialRotation2dLayer((3, 3), (2, 2), no_dc_leakage=True, dtype=DT, angles=1.0, mus=-1)
    # The constructor overwrites the stored values as MATLAB updateParameters does
    assert layer.mus[0].item() == 1
    assert torch.all(layer.angles[:2] == 0)
    layer.set_angles(torch.randn(6, dtype=DT))
    x = torch.zeros(1, 4, 2, 2, dtype=DT)
    x[:, 0] = 1.0
    y = layer(x)
    # A DC input goes only to channel 0
    torch.testing.assert_close(y[:, 0], torch.ones(1, 2, 2, dtype=DT))
    torch.testing.assert_close(y[:, 1:], torch.zeros(1, 5, 2, 2, dtype=DT))
    # The gradients of the angles regarded as zero are not masked (as MATLAB)
    y.sum().backward()
    assert layer.angles.grad.shape == (6,)


@pytest.mark.parametrize(
    "layer, shape",
    [
        (lambda: dcnn.NsoltInitialRotation2dLayer((3, 3), (2, 2), dtype=DT), (2, 4, 3, 3)),
        (lambda: dcnn.NsoltFinalRotation2dLayer((3, 3), (2, 2), dtype=DT), (2, 6, 3, 3)),
        (lambda: dcnn.NsoltIntermediateRotation2dLayer((3, 3), "Analysis", dtype=DT), (2, 6, 3, 3)),
        (lambda: dcnn.NsoltInitialRotation3dLayer((4, 4), (2, 2, 2), dtype=DT), (2, 8, 2, 2, 2)),
        (lambda: dcnn.NsoltIntermediateRotation3dLayer((4, 4), "Synthesis", dtype=DT), (2, 8, 2, 2, 2)),
    ],
)
def test_rotation_gradcheck(layer, shape):
    torch.manual_seed(1)
    m = layer()
    m.set_angles(torch.randn(m.angles.numel(), dtype=DT))
    x = torch.randn(shape, dtype=DT, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x, a: torch.func.functional_call(m, {"angles": a}, (x,)),
        (x, m.angles.detach().clone().requires_grad_()),
    )


def test_mus_in_state_dict_and_device():
    layer = dcnn.NsoltIntermediateRotation2dLayer((3, 3), mus=-1)
    assert set(layer.state_dict()) == {"angles", "mus"}
    torch.testing.assert_close(layer.state_dict()["mus"], -torch.ones(3))


@pytest.mark.gpu
def test_rotation_on_gpu():
    layer = dcnn.NsoltInitialRotation2dLayer((4, 4), (2, 2), no_dc_leakage=True).cuda()
    y = layer(torch.randn(2, 4, 4, 4, device="cuda"))
    assert y.device.type == "cuda"


def test_invalid_arguments():
    with pytest.raises(InvalidMode):
        dcnn.NsoltIntermediateRotation2dLayer((2, 2), mode="Forward")
    with pytest.raises(InvalidMus):
        dcnn.NsoltIntermediateRotation2dLayer((2, 2), mus=[1, 2])
    with pytest.raises(InvalidNumberOfAngles):
        dcnn.NsoltInitialRotation2dLayer((3, 3), (2, 2), angles=[0.0, 1.0])
    with pytest.raises(InvalidDirection):
        dcnn.NsoltAtomExtension2dLayer((3, 3), "Back", "Sum")
    with pytest.raises(InvalidTargetChannels):
        dcnn.NsoltAtomExtension3dLayer((3, 3), "Back", "Both")


@pytest.mark.parametrize("direction", ["Right", "Left", "Down", "Up"])
def test_atom_extension_is_orthogonal(direction):
    for target in ("Sum", "Difference"):
        layer = dcnn.NsoltAtomExtension2dLayer((3, 3), direction, target)
        x = torch.randn(2, 6, 4, 5, dtype=DT)
        assert torch.allclose(layer(x).norm(), x.norm())


def test_channel_separation_concatenation():
    x = torch.randn(2, 6, 3, 4)
    ac, dc = dcnn.NsoltChannelSeparation2dLayer()(x)
    assert ac.shape == (2, 5, 3, 4) and dc.shape == (2, 1, 3, 4)
    torch.testing.assert_close(dcnn.NsoltChannelConcatenation2dLayer()(ac, dc), x)
    parts = dcnn.NsoltComponentSeparation2dLayer(3)(torch.randn(2, 6, 3, 4))
    assert len(parts) == 3 and parts[0].shape == (2, 2, 3, 4)


@pytest.mark.parametrize("ndims", [2, 3])
def test_serialization_round_trip(ndims):
    if ndims == 2:
        srz = dcnn.NsoltSubbandSerialization2dLayer((16, 8), (3, 3), (2, 2), 2)
        dsz = dcnn.NsoltSubbandDeserialization2dLayer((16, 8), (3, 3), (2, 2), 2)
        ins = (torch.randn(2, 5, 8, 4), torch.randn(2, 5, 4, 2), torch.randn(2, 1, 4, 2))
    else:
        srz = dcnn.NsoltSubbandSerialization3dLayer((8, 8, 4), (4, 4), (2, 2, 2), 1)
        dsz = dcnn.NsoltSubbandDeserialization3dLayer((8, 8, 4), (4, 4), (2, 2, 2), 1)
        ins = (torch.randn(2, 7, 2, 4, 4), torch.randn(2, 1, 2, 4, 4))
    z = srz(*ins)
    assert z.shape == (2, srz.number_of_elements)
    for a, b in zip(dsz(z), ins):
        torch.testing.assert_close(a, b)
    # DC of the last level comes first, in column-major order (rows fastest)
    dc = ins[-1][:, 0].reshape(2, -1, *ins[-1].shape[-2:])[:, 0]  # first layer (depth 0)
    torch.testing.assert_close(z[:, 0], dc[:, 0, 0])
    torch.testing.assert_close(z[:, 1], dc[:, 1, 0])
