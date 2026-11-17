"""BoundaryOperation ('Circular' | 'Termination') of the intermediate rotation
layers and the network builders.

The values are compared with MATLAB in test_refdata_layers.py and
test_refdata_networks.py; this file checks the properties that do not need the
reference data, as the MATLAB test classes
nsoltIntermediateRotation2d/3dLayerBoundaryOperationTestCase and
fcn_creatensoltlgraph2dBoundaryOperationTestCase do.
"""

import pytest
import torch

import saivdr.dcnn as dcnn
from saivdr.dcnn import InvalidBoundaryOperation, InvalidDirection
from saivdr.dcnn.matlab_compat import rotation_layers

DT = torch.float64

# Boundary slice of the PyTorch layout: dimension of the spatial axis
# 2-D: nSamples x nChs x nRows x nCols
# 3-D: nSamples x nChs x nLays x nRows x nCols
DIMS2D = {"Vertical": 2, "Horizontal": 3}
DIMS3D = {"Depth": 2, "Vertical": 3, "Horizontal": 4}


def _layer(ndims, nchs, **kw):
    cls = dcnn.NsoltIntermediateRotation2dLayer if ndims == 2 else dcnn.NsoltIntermediateRotation3dLayer
    return cls(nchs, dtype=DT, **kw)


def _random_angles_mus(layer):
    layer.set_angles(torch.randn(layer.angles.numel(), dtype=DT))
    mus = torch.where(torch.rand(layer.mus.numel()) > 0.5, 1.0, -1.0).to(DT)
    layer.set_mus(mus)


def _expected(X, nchs, Un, mode, dim):
    """Rotate the difference channels by Un (Un.T in Synthesis) and by -I on
    the first slice along dim (no boundary slice if dim is None)."""
    ps, pa = nchs
    if mode == "Synthesis":
        Un = Un.T
    Xa = X[:, ps : ps + pa]
    Za = torch.movedim(torch.movedim(Xa, 1, -1) @ Un.T, -1, 1)
    if dim is not None:
        Za = Za.clone()
        Za.narrow(dim, 0, 1).copy_(-Xa.narrow(dim, 0, 1))
    return torch.cat((X[:, :ps], Za), dim=1)


# --- Layers -------------------------------------------------------------------


@pytest.mark.parametrize("ndims", [2, 3])
def test_default_boundary_operation(ndims):
    layer = _layer(ndims, (4, 4))
    assert layer.boundary_operation == "Circular"


@pytest.mark.parametrize("ndims,direction", [(2, d) for d in DIMS2D] + [(3, d) for d in DIMS3D])
def test_constructor_with_boundary_operation(ndims, direction):
    layer = _layer(ndims, (4, 4), boundary_operation="Termination", direction=direction)
    assert layer.boundary_operation == "Termination"
    assert layer.direction == direction


@pytest.mark.parametrize("ndims", [2, 3])
def test_invalid_boundary_operation(ndims):
    with pytest.raises(InvalidBoundaryOperation):
        _layer(ndims, (4, 4), boundary_operation="InvalidBoundaryOperation")
    layer = _layer(ndims, (4, 4))
    with pytest.raises(InvalidBoundaryOperation):
        layer.boundary_operation = "InvalidBoundaryOperation"


@pytest.mark.parametrize("ndims,direction", [(2, "InvalidDirection"), (2, "Depth"), (3, "InvalidDirection")])
def test_invalid_direction(ndims, direction):
    with pytest.raises(InvalidDirection):
        _layer(ndims, (4, 4), boundary_operation="Termination", direction=direction)
    layer = _layer(ndims, (4, 4))
    with pytest.raises(InvalidDirection):
        layer.direction = direction


@pytest.mark.parametrize("ndims", [2, 3])
def test_termination_without_direction(ndims):
    with pytest.raises(InvalidDirection):
        _layer(ndims, (4, 4), boundary_operation="Termination")
    layer = _layer(ndims, (4, 4))
    layer.boundary_operation = "Termination"
    X = torch.zeros((1, 8) + (4,) * ndims, dtype=DT)
    with pytest.raises(InvalidDirection):
        layer(X)


@pytest.mark.parametrize("mode", ["Analysis", "Synthesis"])
@pytest.mark.parametrize(
    "ndims,direction,shape",
    [(2, d, (4, 5)) for d in DIMS2D] + [(3, d, (3, 4, 5)) for d in DIMS3D],
)
def test_termination_boundary_slice(ndims, direction, shape, mode):
    nchs = (3, 3) if ndims == 2 else (4, 4)
    layer = _layer(ndims, nchs, mode=mode, boundary_operation="Termination", direction=direction)
    _random_angles_mus(layer)
    X = torch.randn((2, sum(nchs)) + shape, dtype=DT)
    dims = DIMS2D if ndims == 2 else DIMS3D
    expected = _expected(X, nchs, layer.matrix(), mode, dims[direction])
    torch.testing.assert_close(layer(X), expected)


@pytest.mark.parametrize("ndims,direction,shape", [(2, "Horizontal", (4, 5)), (3, "Depth", (3, 4, 5))])
def test_circular_ignores_direction(ndims, direction, shape):
    nchs = (3, 3) if ndims == 2 else (4, 4)
    layer = _layer(ndims, nchs, mode="Analysis", boundary_operation="Circular", direction=direction)
    _random_angles_mus(layer)
    X = torch.randn((2, sum(nchs)) + shape, dtype=DT)
    torch.testing.assert_close(layer(X), _expected(X, nchs, layer.matrix(), "Analysis", None))


@pytest.mark.parametrize("ndims,direction,shape", [(2, "Vertical", (4, 5)), (3, "Horizontal", (3, 4, 5))])
def test_set_after_construction(ndims, direction, shape):
    nchs = (3, 3) if ndims == 2 else (4, 4)
    layer = _layer(ndims, nchs, mode="Analysis")
    _random_angles_mus(layer)
    layer.direction = direction
    layer.boundary_operation = "Termination"
    X = torch.randn((2, sum(nchs)) + shape, dtype=DT)
    dims = DIMS2D if ndims == 2 else DIMS3D
    expected = _expected(X, nchs, layer.matrix(), "Analysis", dims[direction])
    torch.testing.assert_close(layer(X), expected)


@pytest.mark.parametrize("ndims,direction,shape", [(2, "Horizontal", (4, 5)), (3, "Depth", (3, 4, 5))])
def test_termination_gradcheck(ndims, direction, shape):
    nchs = (2, 2) if ndims == 2 else (4, 4)
    layer = _layer(ndims, nchs, mode="Synthesis", boundary_operation="Termination", direction=direction)
    _random_angles_mus(layer)
    X = torch.randn((1, sum(nchs)) + shape, dtype=DT, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (X,))


# --- Networks -----------------------------------------------------------------

CFG2D = dict(number_of_channels=(3, 3), decimation_factor=(2, 2), polyphase_order=(2, 4))
CFG3D = dict(number_of_channels=(4, 4), decimation_factor=(2, 2, 2), polyphase_order=(2, 2, 2))
LABELS = {"v": "Vertical", "h": "Horizontal", "d": "Depth"}


def _randomize(net):
    torch.manual_seed(0)
    with torch.no_grad():
        for p in net.parameters():
            p.uniform_(-torch.pi, torch.pi)


@pytest.mark.parametrize("ndims", [2, 3])
def test_network_default_boundary_operation(ndims):
    create = dcnn.create_nsolt_2d if ndims == 2 else dcnn.create_nsolt_3d
    net = create(**(CFG2D if ndims == 2 else CFG3D), dtype=DT)
    layers = [lyr for lyr in rotation_layers(net).values() if hasattr(lyr, "boundary_operation")]
    assert layers
    assert all(lyr.boundary_operation == "Circular" for lyr in layers)


@pytest.mark.parametrize("ndims", [2, 3])
@pytest.mark.parametrize("boundary_operation", ["Circular", "Termination"])
def test_network_layer_settings(ndims, boundary_operation):
    create = dcnn.create_nsolt_2d if ndims == 2 else dcnn.create_nsolt_3d
    cfg = CFG2D if ndims == 2 else CFG3D
    net = create(**cfg, boundary_operation=boundary_operation, dtype=DT)
    layers = {n: lyr for n, lyr in rotation_layers(net).items() if hasattr(lyr, "boundary_operation")}
    assert len(layers) == 2 * sum(cfg["polyphase_order"])
    for name, layer in layers.items():
        label, order = name.rstrip("~").split("_V")[-1][0], int(name.rstrip("~").split("_V")[-1][1:])
        if order % 2 == 1:
            assert layer.boundary_operation == boundary_operation, name
            assert layer.direction == LABELS[label], name
        else:
            assert layer.boundary_operation == "Circular", name


def test_network_invalid_boundary_operation():
    with pytest.raises(InvalidBoundaryOperation):
        dcnn.create_nsolt_2d(**CFG2D, boundary_operation="InvalidBoundaryOperation")


@pytest.mark.parametrize("ndims", [2, 3])
@pytest.mark.parametrize("nlevels", [1, 2])
def test_network_perfect_reconstruction(ndims, nlevels):
    create = dcnn.create_nsolt_2d if ndims == 2 else dcnn.create_nsolt_3d
    cfg = CFG2D if ndims == 2 else CFG3D
    analyzer = create(
        **cfg, number_of_levels=nlevels, mode="Analyzer", boundary_operation="Termination", dtype=DT
    )
    _randomize(analyzer)
    synthesizer = analyzer.T
    syn_layers = [lyr for lyr in rotation_layers(synthesizer).values() if hasattr(lyr, "boundary_operation")]
    assert any(lyr.boundary_operation == "Termination" for lyr in syn_layers)
    shape = (24, 16) if ndims == 2 else (8, 24, 16)
    X = torch.randn((2, 1) + shape, dtype=DT)
    with torch.no_grad():
        torch.testing.assert_close(synthesizer(*analyzer(X)), X, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("boundary_operation", ["Circular", "Termination"])
def test_copy_params_boundary_operation(boundary_operation):
    # The analyzer created with the other mode becomes the adjoint after copying
    other = "Termination" if boundary_operation == "Circular" else "Circular"
    synthesizer = dcnn.create_nsolt_2d(
        **CFG2D, mode="Synthesizer", boundary_operation=boundary_operation, dtype=DT
    )
    _randomize(synthesizer)
    analyzer = dcnn.create_nsolt_2d(**CFG2D, mode="Analyzer", boundary_operation=other, dtype=DT)
    dcnn.copy_params_syn2ana(analyzer, synthesizer)
    syn_layers = rotation_layers(synthesizer)
    for name, alayer in rotation_layers(analyzer).items():
        if hasattr(alayer, "boundary_operation"):
            assert alayer.boundary_operation == syn_layers[name + "~"].boundary_operation, name
            assert alayer.direction == syn_layers[name + "~"].direction, name
    X = torch.randn(2, 1, 24, 16, dtype=DT)
    with torch.no_grad():
        torch.testing.assert_close(synthesizer(*analyzer(X)), X, rtol=1e-10, atol=1e-10)
    # and back from the analyzer to a synthesizer created with the other mode
    synthesizer2 = dcnn.create_nsolt_2d(**CFG2D, mode="Synthesizer", boundary_operation=other, dtype=DT)
    dcnn.copy_params_ana2syn(synthesizer2, analyzer)
    with torch.no_grad():
        torch.testing.assert_close(synthesizer2(*analyzer(X)), X, rtol=1e-10, atol=1e-10)


def _edge_responses(ndims, boundary_operation, dim):
    """Max |coefs| on the last block along dim for an input on the first block."""
    create = dcnn.create_nsolt_2d if ndims == 2 else dcnn.create_nsolt_3d
    cfg = CFG2D if ndims == 2 else CFG3D
    analyzer = create(**cfg, mode="Analyzer", boundary_operation=boundary_operation, dtype=DT)
    _randomize(analyzer)
    shape = (16, 16) if ndims == 2 else (16, 16, 16)
    X = torch.zeros((1, 1) + shape, dtype=DT)
    X.narrow(dim, 0, 2).normal_()  # decimation factor 2 in every direction
    with torch.no_grad():
        coefs = analyzer(X)
    return max(c.narrow(dim, c.shape[dim] - 1, 1).abs().max().item() for c in coefs)


@pytest.mark.parametrize("ndims,dim", [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4)])
def test_network_no_wrap_around(ndims, dim):
    assert _edge_responses(ndims, "Termination", dim) == pytest.approx(0.0, abs=1e-10)
    assert _edge_responses(ndims, "Circular", dim) > 1e-4
