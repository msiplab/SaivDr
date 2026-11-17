"""Noise processes (MATLAB saivdr.degradation.noiseprocess)."""

import torch

__all__ = ["AdditiveWhiteGaussianNoiseSystem"]


class AdditiveWhiteGaussianNoiseSystem:
    """Additive white Gaussian noise, equivalent to MATLAB ``imnoise(x,'gaussian',mean,variance)``.

    As ``imnoise`` for double images, the result is clipped to [0, 1].
    ``generator`` (a :class:`torch.Generator` on any device) makes the noise
    reproducible.
    """

    def __init__(self, mean=0.0, variance=0.01, generator=None):
        self.mean = mean
        self.variance = variance
        self.generator = generator

    def __call__(self, x):
        # Draw the noise on the generator's device, which may differ from x's
        device = x.device if self.generator is None else self.generator.device
        noise = torch.randn(x.shape, generator=self.generator, dtype=x.dtype, device=device)
        return (x + self.mean + noise.to(x.device) * self.variance**0.5).clamp(0.0, 1.0)
