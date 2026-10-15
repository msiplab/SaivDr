"""Degradation processes ported from MATLAB ``+saivdr/+degradation`` (2-D images)."""

from .linearprocess import (
    AbstLinearSystem,
    BlurSystem,
    DecimationSystem,
    PixelLossSystem,
    fspecial_average,
    fspecial_gaussian,
    imfilter,
)
from .noiseprocess import AdditiveWhiteGaussianNoiseSystem

__all__ = [
    "AbstLinearSystem",
    "AdditiveWhiteGaussianNoiseSystem",
    "BlurSystem",
    "DecimationSystem",
    "DegradationSystem",
    "PixelLossSystem",
    "fspecial_average",
    "fspecial_gaussian",
    "imfilter",
]


class DegradationSystem:
    """Linear process followed by a noise process (MATLAB ``DegradationSystem``)."""

    def __init__(self, linear_process, noise_process):
        self.linear_process = linear_process
        self.noise_process = noise_process

    def __call__(self, x):
        return self.noise_process(self.linear_process.forward(x))
