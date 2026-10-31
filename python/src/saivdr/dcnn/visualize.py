"""Atomic images of NSOLT synthesizers (MATLAB saivdr.dcnn.atomicimshow)."""

import math

import torch

__all__ = ["atomic_images", "atomic_imshow"]


def _synthesizer(net):
    for attr in ("synthesizer",):
        if hasattr(net, attr):
            return getattr(net, attr)
    return net


def _matlab_round_half(n):
    """0-based index of ``round(end/2)`` in MATLAB."""
    return int(math.floor(n / 2 + 0.5)) - 1


def default_patch_size(synthesizer, margin=2):
    """Patch size large enough to hold the atoms of ``synthesizer``.

    MATLAB ``atomicimshow`` estimates the polyphase order from the number of
    layers; here the actual polyphase order is used instead.
    """
    dec = synthesizer.decimation_factor
    ppo = synthesizer.polyphase_order
    nLevels = synthesizer.number_of_levels
    ext = [d * (o + 1) for d, o in zip(dec, ppo)]
    for _ in range(2, nLevels + 1):
        ext = [(e - 1) * (d + 1) + 1 for e, d in zip(ext, dec)]
    maxdec = [d**nLevels for d in dec]
    return tuple((math.ceil(e / m) + margin) * m for e, m in zip(ext, maxdec))


@torch.no_grad()
def atomic_images(synthesizer, patch_size=None):
    """Return the atomic images (impulse responses) of a 2-D synthesizer.

    Equivalent to the images that MATLAB ``atomicimshow`` displays. The
    result is a tensor of ``nChsTotal x nComponents x height x width``, where
    the first atom is the DC atom of the last level, followed by the AC atoms
    of levels ``nLevels, ..., 1``. An impulse is placed at index
    ``round(end/2)`` (MATLAB) of each subband.
    """
    syn = _synthesizer(synthesizer)
    if patch_size is None:
        patch_size = default_patch_size(syn)
    patch_size = tuple(int(v) for v in patch_size)
    dec = syn.decimation_factor
    nLevels = syn.number_of_levels
    nComponents = syn.number_of_components
    nChsPerLv = sum(syn.number_of_channels)
    param = next(syn.parameters())
    dtype, device = param.dtype, param.device

    def zeros(iLv, nchs):
        size = [p // d**iLv for p, d in zip(patch_size, dec)]
        return torch.zeros((1, nchs, *size), dtype=dtype, device=device)

    base = [zeros(iLv, nComponents * (nChsPerLv - 1)) for iLv in range(1, nLevels + 1)]
    base.append(zeros(nLevels, nComponents))

    def impulse(inputs, index, channels):
        x = inputs[index].clone()
        r = _matlab_round_half(x.size(2))
        c = _matlab_round_half(x.size(3))
        x[0, channels, r, c] = 1
        out = list(inputs)
        out[index] = x
        return out

    atoms = [syn(*impulse(base, nLevels, list(range(nComponents))))[0]]
    for iRevLv in range(nLevels, 0, -1):
        for iAtom in range(nChsPerLv - 1):
            channels = [iCmp * (nChsPerLv - 1) + iAtom for iCmp in range(nComponents)]
            atoms.append(syn(*impulse(base, iRevLv - 1, channels))[0])
    return torch.stack(atoms, dim=0)


def atomic_imshow(synthesizer, patch_size=None, scale=1.0, ax=None):
    """Display the atomic images as a montage, as MATLAB ``atomicimshow``.

    Requires matplotlib. Returns the matplotlib image.
    """
    import matplotlib.pyplot as plt

    atoms = atomic_images(synthesizer, patch_size).cpu()
    nAtoms, nComponents, height, width = atoms.shape
    mRows = 2 ** (math.ceil(math.log2(math.sqrt(nAtoms))) - 1) if nAtoms > 1 else 1
    mCols = math.ceil(nAtoms / mRows)
    border = 2
    zoom = 8
    canvas = torch.ones(
        (mRows * (height * zoom + border) + border, mCols * (width * zoom + border) + border, nComponents)
    )
    for i in range(nAtoms):
        r, c = divmod(i, mCols)
        img = (scale * atoms[i] + 0.5).clamp(0, 1).permute(1, 2, 0)
        img = img.repeat_interleave(zoom, 0).repeat_interleave(zoom, 1)
        y = border + r * (height * zoom + border)
        x = border + c * (width * zoom + border)
        canvas[y : y + height * zoom, x : x + width * zoom] = img
    if ax is None:
        ax = plt.gca()
    ax.axis("off")
    return ax.imshow(canvas.squeeze(-1).numpy(), cmap="gray" if nComponents == 1 else None, vmin=0, vmax=1)
