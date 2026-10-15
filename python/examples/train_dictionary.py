"""Dictionary learning of an NSOLT synthesizer (cf. examples/dcnn/main_nsolt_diclrn_uni.m).

Alternates the sparse approximation of image patches by iterative hard
thresholding (IHT) with the analyzer as the adjoint, and the update of the
synthesizer by SGD with momentum, as the MATLAB example does.

    python train_dictionary.py [--image barbara.png] [--device cuda] [--output atoms.png]
"""

import argparse
import math

import torch

from saivdr.dcnn import copy_params_syn2ana, create_nsolt_2d


def load_image(path):
    if path is None:
        # Synthetic image: smooth gradients and oriented stripes
        y, x = torch.meshgrid(torch.linspace(0, 1, 256), torch.linspace(0, 1, 256), indexing="ij")
        return 0.5 + 0.25 * torch.sin(40 * (x + 0.5 * y)) * (x > 0.5) + 0.25 * (y - 0.5)
    import matplotlib.image as mpimg

    img = torch.as_tensor(mpimg.imread(path), dtype=torch.float32)
    if img.ndim == 3:
        img = img[..., :3].mean(-1)
    return img / img.max()


def random_patches(img, size, n, generator):
    h, w = img.shape
    rows = torch.randint(0, h - size[0] + 1, (n,), generator=generator)
    cols = torch.randint(0, w - size[1] + 1, (n,), generator=generator)
    return torch.stack([img[r : r + size[0], c : c + size[1]] for r, c in zip(rows, cols)]).unsqueeze(1)


@torch.no_grad()
def iht(x, analyzer, synthesizer, sparsity_ratio, n_iters=10, gamma=1 - 1e-3):
    """Iterative hard thresholding keeping ``sparsity_ratio`` of the coefficients per sample."""
    n_samples = x.size(0)
    n_coefs = math.floor(sparsity_ratio * x[0].numel())
    coefs = analyzer(torch.zeros_like(x))
    for _ in range(n_iters):
        grad = analyzer(synthesizer(*coefs) - x)
        coefs = tuple(c - gamma * g for c, g in zip(coefs, grad))
        flat = torch.cat([c.reshape(n_samples, -1) for c in coefs], dim=1).abs()
        thk = flat.sort(dim=1, descending=True).values[:, n_coefs - 1]
        coefs = tuple(c * (c.abs() > thk.view(-1, *([1] * (c.dim() - 1)))) for c in coefs)
    return coefs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output", default=None, help="save the atomic images (requires matplotlib)")
    args = parser.parse_args()

    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    cfg = dict(
        number_of_channels=(3, 3),
        decimation_factor=(2, 2),
        polyphase_order=(2, 2),
        number_of_vanishing_moments=1,
        device=args.device,
    )
    patch_size = (32, 32)
    analyzer = create_nsolt_2d(mode="Analyzer", **cfg)
    synthesizer = create_nsolt_2d(mode="Synthesizer", **cfg)
    with torch.no_grad():
        for p in synthesizer.parameters():
            p.add_(torch.randn_like(p) * math.pi / 6)
    copy_params_syn2ana(analyzer, synthesizer)

    patches = random_patches(load_image(args.image), patch_size, 64, gen).to(args.device)
    with torch.no_grad():
        mse = (synthesizer(*analyzer(patches)) - patches).pow(2).mean()
    print(f"Initial MSE of perfect reconstruction: {mse:.3e}")
    optimizer = torch.optim.SGD(synthesizer.parameters(), lr=0.01, momentum=0.9)
    for it in range(args.iterations):
        coefs = iht(patches, analyzer, synthesizer, sparsity_ratio=1 / 3)
        for _ in range(args.epochs):
            perm = torch.randperm(patches.size(0), generator=gen)
            for idx in perm.split(8):
                optimizer.zero_grad()
                loss = 0.5 * (synthesizer(*(c[idx] for c in coefs)) - patches[idx]).pow(2).mean()
                loss.backward()
                optimizer.step()
        copy_params_syn2ana(analyzer, synthesizer)
        with torch.no_grad():
            coefs = iht(patches, analyzer, synthesizer, sparsity_ratio=1 / 3)
            mse = (synthesizer(*coefs) - patches).pow(2).mean()
        print(f"Iteration {it + 1}: MSE of sparse approximation {mse:.3e}")

    if args.output:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from saivdr.dcnn import atomic_imshow

        atomic_imshow(synthesizer.cpu())
        plt.title("Atomic images of trained NSOLT")
        plt.savefig(args.output, bbox_inches="tight")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
