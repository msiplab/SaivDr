"""Compare the atomic images with MATLAB atomicimshow (refdata/atoms)."""

import pytest
import torch

from _refdata import array, case_files, kwargs, load, params_dict, text
from saivdr.dcnn import atomic_images, create_nsolt_2d
from saivdr.dcnn.matlab_compat import set_params

FILES = case_files("atoms")


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_atoms(path):
    m = load(path)
    syn = create_nsolt_2d(dtype=torch.float64, **kwargs(text(m, "meta_params")))
    set_params(syn, params_dict(m))
    atoms = atomic_images(syn, patch_size=m["patchsize"].ravel())
    expected = array(m, "atoms", "image2d")  # H x W x nCmps x nAtoms -> nAtoms x nCmps x H x W
    torch.testing.assert_close(atoms, expected, rtol=1e-9, atol=1e-10)
