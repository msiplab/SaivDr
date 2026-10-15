import math

import pytest
import torch

from saivdr.dcnn import InvalidMus, orthmtxgen
from saivdr.dcnn._orthmtx import check_mus, number_of_dimensions


def givens(n, i, j, angle):
    G = torch.eye(n, dtype=torch.float64)
    c, s = math.cos(angle), math.sin(angle)
    G[i, i], G[i, j], G[j, i], G[j, j] = c, -s, s, c
    return G


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
def test_matches_explicit_givens_product(n):
    torch.manual_seed(n)
    angles = torch.randn(n * (n - 1) // 2, dtype=torch.float64)
    mus = torch.where(torch.rand(n) < 0.5, -1.0, 1.0).double()
    expected = torch.eye(n, dtype=torch.float64)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            expected = givens(n, i, j, angles[k].item()) @ expected
            k += 1
    expected = torch.diag(mus) @ expected
    torch.testing.assert_close(orthmtxgen(angles, mus), expected, rtol=0, atol=1e-14)


def test_orthonormal_and_scalar_mus():
    angles = torch.randn(28, dtype=torch.float64)
    M = orthmtxgen(angles, -1)
    torch.testing.assert_close(M @ M.T, torch.eye(8, dtype=torch.float64), rtol=0, atol=1e-14)


def test_gradcheck():
    angles = torch.randn(6, dtype=torch.float64, requires_grad=True)
    mus = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda a: orthmtxgen(a, mus), (angles,))


def test_invalid():
    with pytest.raises(ValueError):
        number_of_dimensions(4)
    with pytest.raises(InvalidMus):
        check_mus(torch.tensor([1.0, 0.5]))
