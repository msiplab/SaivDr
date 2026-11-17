"""ISTA-based image restoration (MATLAB saivdr.restoration.ista.IstaImRestoration2d)."""

import math

import torch

__all__ = ["IstaImRestoration2d", "softshrink", "psnr"]


def softshrink(x, threshold):
    """Soft thresholding, as ``AbstIstaImRestoration.softshrink_`` in MATLAB."""
    nc = x.abs() - threshold
    nc = (nc + nc.abs()) / 2
    return torch.sign(x) * nc


def psnr(x, ref, peak=1.0):
    """Peak signal-to-noise ratio in dB, as MATLAB ``psnr`` for double images."""
    mse = torch.mean(
        (torch.as_tensor(x, dtype=torch.float64) - torch.as_tensor(ref, dtype=torch.float64)) ** 2
    )
    return float(10 * torch.log10(peak**2 / mse))


class IstaImRestoration2d:
    """ISTA-based image restoration, equivalent to MATLAB ``IstaImRestoration2d``.

    Solves ``min_y 1/2 ||x - P D y||^2 + lambda ||y||_1`` by the iterative
    shrinkage/thresholding algorithm, where ``D`` is the synthesizer and
    ``P`` the linear process, and returns ``D y``.

    Parameters
    ----------
    synthesizer, adj_of_synthesizer : torch.nn.Module
        Single-component synthesizer and its adjoint (analyzer), e.g. the
        networks of :func:`saivdr.dcnn.create_nsolt_2d`. Each component
        (channel) of the image is processed with them independently.
    linear_process : saivdr.degradation.AbstLinearSystem
    lambda_ : float
        Regularization parameter.
    eps0 : float
        Tolerance of ``||y_k - y_{k-1}||^2 / ||y_k||^2``.
    max_iter : int
    frame_bound : float
        Upper frame bound of the synthesizer (1 for a Parseval tight frame).
    step_monitor : callable, optional
        Called as ``step_monitor(n_iter, restored)`` after every iteration.
    lambda_max : float, optional
        Maximum eigenvalue of ``P^T P``; estimated by the power method if omitted.
    """

    def __init__(
        self,
        synthesizer,
        adj_of_synthesizer,
        linear_process,
        lambda_,
        eps0=1e-6,
        max_iter=1000,
        frame_bound=1.0,
        step_monitor=None,
        lambda_max=None,
    ):
        self.synthesizer = synthesizer
        self.adj_of_synthesizer = adj_of_synthesizer
        self.linear_process = linear_process
        self.lambda_ = lambda_
        self.eps0 = eps0
        self.max_iter = max_iter
        self.frame_bound = frame_bound
        self.step_monitor = step_monitor
        self.lambda_max = lambda_max
        self.n_iter = 0
        self.err = math.inf

    def _analyze(self, x):
        shape = x.shape
        return self.adj_of_synthesizer(x.reshape(-1, 1, *shape[-2:])), shape

    def _synthesize(self, y, shape):
        return self.synthesizer(*y).reshape(shape)

    @torch.no_grad()
    def __call__(self, observed):
        """Restore ``observed`` (a tensor whose last two dimensions are spatial)."""
        x = observed
        lin = self.linear_process
        lmax = self.lambda_max
        if lmax is None:
            lmax = lin.lambda_max(x.shape[-2:], dtype=x.dtype, device=x.device)
        valueL = self.frame_bound * lmax
        hu = lin.adjoint(x)
        y, shape = self._analyze(hu)
        r = lin.forward(hu) - x
        threshold = self.lambda_ / valueL
        self.n_iter = 0
        self.err = math.inf
        ypre = y
        while self.err > self.eps0 and self.n_iter < self.max_iter:
            self.n_iter += 1
            v, _ = self._analyze(lin.adjoint(r))
            y = tuple(softshrink(c - v_ / valueL, threshold) for c, v_ in zip(y, v))
            hu = self._synthesize(y, shape)
            r = lin.forward(hu) - x
            num = sum(((a - b) ** 2).sum() for a, b in zip(y, ypre))
            den = sum((a**2).sum() for a in y)
            self.err = float(num / den)
            ypre = y
            if self.step_monitor is not None:
                self.step_monitor(self.n_iter, hu)
        return hu
